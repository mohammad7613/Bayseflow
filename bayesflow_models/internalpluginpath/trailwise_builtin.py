from bayesflow_models.interfaces import ModelSpec
from bayesflow_models.DDM_DC_Pedestrain_TrialWise import (
      model_DC_TrialWise,
      model_DC,
      get_adapter,
      get_adapter_trialwise,
      PAR_NAMES
  )

from bayesflow_models.reproduce_models import (model_DF_reproduced,
                                               model_DC_reproduced,
                                               model_SF_reproduced,
                                               PAR_NAMES_DC_REPRODUCED,
                                               PAR_NAMES_DF_REPRODUCED,
                                               PAR_NAMES_SF_REPRODUCED,
                                               get_adapter_trialwise_dc,
                                               get_adapter_trialwise_df,
                                               get_adapter_trialwise_sf)

MODEL_SPECS = [
    # ModelSpec(
    #     name="model_DC_TwoBoundary_Simplest",
    #     workflow="builtin",
    #     simulator_factory=lambda: model_DC,
    #     adapter_factory=get_adapter,
    #     family="ddm_dc",
    #     par_names = PAR_NAMES,
    #     version="1.0",
    #     description="Built-in simplest DDM with the whole TTA in the simulator",
    # ),
    #     ModelSpec(
    #     name="model_DC_TwoBoundary_TiralWise_Simplest",
    #     workflow="builtin",
    #     simulator_factory=lambda: model_DC_TrialWise,
    #     adapter_factory=get_adapter_trialwise,
    #     family="ddm_dc",
    #     par_names = PAR_NAMES,
    #     version="1.0",
    #     description="Built-in simplest DDM with TiralWise TA conditions in the simulator",
    # ),       
    ModelSpec(
        name="model_DC",
        workflow="builtin",
        simulator_factory=lambda: model_DC_reproduced,
        adapter_factory=get_adapter_trialwise_dc,
        family="ddm",
        par_names = PAR_NAMES_DC_REPRODUCED,
        version="1.0",
        description="This is ddm models of the main article with dynamic drift rate and collapsing boundary",
    ),
    ModelSpec(
        name="model_DF",
        workflow="builtin",
        simulator_factory=lambda: model_DF_reproduced,
        adapter_factory=get_adapter_trialwise_df,
        family="ddm",
        par_names = PAR_NAMES_DF_REPRODUCED,
        version="1.0",
        description="This is ddm models of the main article with dynamic drift rate and fixed boundary",
    ),
    ModelSpec(
        name="model_SF",
        workflow="builtin",
        simulator_factory=lambda: model_SF_reproduced,
        adapter_factory=get_adapter_trialwise_sf,
        family="ddm",
        par_names = PAR_NAMES_SF_REPRODUCED,
        version="1.0",
        description="This is ddm models of the main article with static drift rate and fixed boundary",
    )
]