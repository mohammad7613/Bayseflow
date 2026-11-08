# In this 
from bayesflow_models.workflow import train_from_spec, resume_from_artifact, recovery_from_artifact
from bayesflow_models.interfaces import Workflow


# This is the test variable define to test discovery from bultin-in path
WORKFLOWS : list = [Workflow(name="builtinExternal", train_fn=train_from_spec, resume_fn=resume_from_artifact, recovery_fn=recovery_from_artifact)]



