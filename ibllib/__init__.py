"""Package used to house a variety of commonly used IBL functionality. A major component of the
package is the data extraction pipelines for modalities.

Instructions to create a new modality:
- Ensure a new task version and new dataset types have already been created.
- Add a new function to ibllib.pipes.misc for copying the raw data to the local server. Call this
  function from within confirm_ephys_remote_folder if the data are acquired on the ephys PC, or
  confirm_video_remote_folder if the data are on the video PC. If the data are on an entirely
  different PC, create a new function.
- Add a user script to iblscripts.deploy that calls the above function, so that users can run it
  from the command line.
- In ibllib.oneibl.registration add a pattern to the REGISTRATION_GLOB_PATTERNS variable if
  necessary.
- Add a new procedure to the _alyx_procedure_from_task_type function if necessary.
- Add a new extractor type to the ibllib/io/extractors/extractor_types.json file.
- Add a new pipeline name to ibllib.io.extractors.base._get_pipeline_from_task_type
- In ibllib.pipes.local_server._get_pipeline_class, add the new pipeline name to the if/else
  statement. Once a new pipeline class has been created, it will be instantiated and returned here.
- Create a new module in ibllib.pipes that will contain your new Pipeline subclass and
  modality-specific tasks.
- In this module import any tasks from other pipelines that can be re-used. Add your new tasks
  here.
"""
__version__ = "2.10.5"
import logging
import warnings

from ibllib.misc import logger_config

warnings.filterwarnings("always", category=DeprecationWarning, module="ibllib")

# if this becomes a full-blown library we should let the logging configuration to the discretion of
# the dev who uses the library. However since it can also be provided as an app, the end-users
# should be provided with a useful default logging in standard output without messing with the
# complex python logging system

USE_LOGGING = True
# %(asctime)s,%(msecs)d
if USE_LOGGING:
    logger_config(name="ibllib")
else:
    # deactivate all log calls for use as a library
    logging.getLogger("ibllib").addHandler(logging.NullHandler())

try:
    import one
except ModuleNotFoundError:
    logging.getLogger("ibllib").error("Missing dependency, please run `pip install ONE-api`")
