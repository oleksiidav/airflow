# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from __future__ import annotations

import logging
import os
import time

import pytest as pytest

from airflow import AirflowException
from airflow.jobs.job import Job, run_job
from airflow.jobs.local_task_job_runner import LocalTaskJobRunner
from airflow.listeners.listener import get_listener_manager
from airflow.models import DagBag, TaskInstance
from airflow.operators.bash import BashOperator
from airflow.task.task_runner.standard_task_runner import StandardTaskRunner
from airflow.utils import timezone
from airflow.utils.session import provide_session
from airflow.utils.state import DagRunState, State, TaskInstanceState
from airflow.utils.timeout import timeout
from tests.listeners import (
    class_listener,
    full_listener,
    lifecycle_listener,
    partial_listener,
    throwing_listener,
    xcom_listener,
)
from tests.models import DEFAULT_DATE
from tests.utils.test_helpers import MockJobRunner

LISTENERS = [
    class_listener,
    full_listener,
    lifecycle_listener,
    partial_listener,
    throwing_listener,
]

DAG_ID = "test_listener_dag"
TASK_ID = "test_listener_task"
EXECUTION_DATE = timezone.utcnow()

TEST_DAG_FOLDER = os.environ["AIRFLOW__CORE__DAGS_FOLDER"]


@pytest.fixture(autouse=True)
def clean_listener_manager():
    lm = get_listener_manager()
    lm.clear()
    yield
    lm = get_listener_manager()
    lm.clear()
    for listener in LISTENERS:
        listener.clear()


@provide_session
def test_listener_gets_calls(create_task_instance, session=None):
    lm = get_listener_manager()
    lm.add_listener(full_listener)

    ti = create_task_instance(session=session, state=TaskInstanceState.QUEUED)
    # Using ti.run() instead of ti._run_raw_task() to capture state change to RUNNING
    # that only happens on `check_and_change_state_before_execution()` that is called before
    # `run()` calls `_run_raw_task()`
    ti.run()

    assert len(full_listener.state) == 2
    assert full_listener.state == [TaskInstanceState.RUNNING, TaskInstanceState.SUCCESS]


@provide_session
def test_multiple_listeners(create_task_instance, session=None):
    lm = get_listener_manager()
    lm.add_listener(full_listener)
    lm.add_listener(lifecycle_listener)
    class_based_listener = class_listener.ClassBasedListener()
    lm.add_listener(class_based_listener)

    job = Job()
    job_runner = MockJobRunner(job=job)
    try:
        run_job(job=job, execute_callable=job_runner._execute)
    except NotImplementedError:
        pass  # just for lifecycle

    assert full_listener.started_component is job
    assert lifecycle_listener.started_component is job
    assert full_listener.stopped_component is job
    assert lifecycle_listener.stopped_component is job
    assert class_based_listener.state == [DagRunState.RUNNING, DagRunState.SUCCESS]


@provide_session
def test_listener_gets_only_subscribed_calls(create_task_instance, session=None):
    lm = get_listener_manager()
    lm.add_listener(partial_listener)

    ti = create_task_instance(session=session, state=TaskInstanceState.QUEUED)
    # Using ti.run() instead of ti._run_raw_task() to capture state change to RUNNING
    # that only happens on `check_and_change_state_before_execution()` that is called before
    # `run()` calls `_run_raw_task()`
    ti.run()

    assert len(partial_listener.state) == 1
    assert partial_listener.state == [TaskInstanceState.RUNNING]


@provide_session
def test_listener_throws_exceptions(create_task_instance, session=None):
    lm = get_listener_manager()
    lm.add_listener(throwing_listener)

    ti = create_task_instance(session=session, state=TaskInstanceState.QUEUED)
    with pytest.raises(RuntimeError):
        ti._run_raw_task()


@provide_session
def test_listener_captures_failed_taskinstances(create_task_instance_of_operator, session=None):
    lm = get_listener_manager()
    lm.add_listener(full_listener)

    ti = create_task_instance_of_operator(
        BashOperator, dag_id=DAG_ID, execution_date=EXECUTION_DATE, task_id=TASK_ID, bash_command="exit 1"
    )
    with pytest.raises(AirflowException):
        ti._run_raw_task()

    assert full_listener.state == [TaskInstanceState.RUNNING, TaskInstanceState.FAILED]
    assert len(full_listener.state) == 2


@provide_session
def test_listener_captures_longrunning_taskinstances(create_task_instance_of_operator, session=None):
    lm = get_listener_manager()
    lm.add_listener(full_listener)

    ti = create_task_instance_of_operator(
        BashOperator, dag_id=DAG_ID, execution_date=EXECUTION_DATE, task_id=TASK_ID, bash_command="sleep 5"
    )
    ti._run_raw_task()

    assert full_listener.state == [TaskInstanceState.RUNNING, TaskInstanceState.SUCCESS]
    assert len(full_listener.state) == 2


@provide_session
def test_class_based_listener(create_task_instance, session=None):
    lm = get_listener_manager()
    listener = class_listener.ClassBasedListener()
    lm.add_listener(listener)

    ti = create_task_instance(session=session, state=TaskInstanceState.QUEUED)
    # Using ti.run() instead of ti._run_raw_task() to capture state change to RUNNING
    # that only happens on `check_and_change_state_before_execution()` that is called before
    # `run()` calls `_run_raw_task()`
    ti.run()

    assert len(listener.state) == 2
    assert listener.state == [TaskInstanceState.RUNNING, TaskInstanceState.SUCCESS]


def test_ol_does_not_block_xcoms():
    """
    Test that ensures that where a task is marked success in the UI
    on_success_callback gets executed
    """

    path_listener_writer = "/tmp/test_ol_does_not_block_xcoms"
    try:
        os.unlink(path_listener_writer)
    except OSError:
        pass

    listener = xcom_listener.XComListener(path_listener_writer, "push_and_pull")
    get_listener_manager().add_listener(listener)
    log = logging.getLogger("airflow")

    dagbag = DagBag(
        dag_folder=TEST_DAG_FOLDER,
        include_examples=False,
    )
    dag = dagbag.dags.get("test_dag_xcom_openlineage")
    task = dag.get_task("push_and_pull")
    dag.create_dagrun(
        run_id="test",
        data_interval=(DEFAULT_DATE, DEFAULT_DATE),
        state=State.RUNNING,
        start_date=DEFAULT_DATE,
    )

    ti = TaskInstance(task=task, run_id="test")
    job = Job(dag_id=ti.dag_id)
    job_runner = LocalTaskJobRunner(job=job, task_instance=ti, ignore_ti_state=True)
    task_runner = StandardTaskRunner(job_runner)
    task_runner.start()

    # Wait until process makes itself the leader of its own process group
    with timeout(seconds=1):
        while True:
            runner_pgid = os.getpgid(task_runner.process.pid)
            if runner_pgid == task_runner.process.pid:
                break
            time.sleep(0.01)

    # Wait till process finishes
    assert task_runner.return_code(timeout=10) is not None
    log.error(task_runner.return_code())

    with open(path_listener_writer) as f:
        assert f.readline() == "on_task_instance_running\n"
        assert f.readline() == "on_task_instance_success\n"
        assert f.readline() == "listener\n"
