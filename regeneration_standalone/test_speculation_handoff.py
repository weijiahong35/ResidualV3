"""The rolling handoff must publish real generation before allowing new GPU claims."""
from contextlib import ExitStack
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0,str(Path(__file__).resolve().parent/'open_perfectblend_regen'))
import tune_speculation as tuning


class HandoffChecks(unittest.TestCase):
    def test_scheduler_stays_paused_until_four_conversations_are_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);worker=root/'worker';worker.mkdir()
            profile=SimpleNamespace(name='nonthinking_ctx4096',scheduler_dir=root/'scheduler',
                worker_dir=lambda index:worker)
            session=tuning.Session(profile,6,root,Mock())
            session.uuid='GPU-test';session.port=18086
            session.old={'pid':30};session.guard=Mock(pid=20)
            child=Mock(pid=40);child.poll.return_value=None
            events=[]
            progress=[0,2,4]
            scheduler={'pid':10}
            def send(record,sig):events.append((record['pid'],sig))
            def read(path):
                if path.parent==profile.scheduler_dir:return {'scheduler_pid':10}
                self.assertIn((10,signal.SIGSTOP),events)
                self.assertNotIn((10,signal.SIGCONT),events)
                return {'state':'generating','supervisor_pid':40,'server_pid':50,
                    'num_speculative_tokens':5,'client_progress':{'ok':progress[0],'errors':0}}
            def wait(predicate,timeout,description):
                for _ in range(4):
                    if predicate():return
                    progress.pop(0)
                self.fail(description)
            with ExitStack() as stack:
                for context in [patch.object(tuning,'read',side_effect=read),
                    patch.object(tuning,'identity',return_value=scheduler),
                    patch.object(tuning,'send',side_effect=send),
                    patch.object(tuning,'stopped',return_value=True),
                    patch.object(tuning,'matching',return_value=False),
                    patch.object(tuning,'wait_for',side_effect=wait),
                    patch.object(tuning,'detached',return_value=child)]:stack.enter_context(context)
                state=session.resume()
            self.assertEqual(state['client_progress']['ok'],4)
            self.assertEqual(events[-1],(10,signal.SIGCONT))
            session.save.assert_called_once()


if __name__=='__main__':unittest.main()
