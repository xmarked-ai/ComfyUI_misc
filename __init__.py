from .ColorGrade         import GRADE_CLASS_MAPPINGS, GRADE_NAME_MAPPINGS
from .RelightX           import RELIGHT_CLASS_MAPPINGS, RELIGHT_NAME_MAPPINGS
from .TileImageSquare    import TILE_CLASS_MAPPINGS, TILE_NAME_MAPPINGS
from .DepthDisplaceX     import DEPTHDISPLACE_CLASS_MAPPINGS, DEPTHDISPLACE_NAME_MAPPINGS
from .NF4_bnb_loaderX    import NF4BNB_CLASS_MAPPINGS, NF4BNB_NAME_MAPPINGS
from .AceColorFixX       import ACECOLORFIXX_CLASS_MAPPINGS, ACECOLORFIXX_NAME_MAPPINGS
from .LoraBatchX         import LORAX_CLASS_MAPPINGS, LORAX_NAME_MAPPINGS
from .KSamplerComboX     import KSAMPLERCOMBOX_CLASS_MAPPINGS, KSAMPLERCOMBOX_NAME_MAPPINGS
from .EmptyLatentX       import EMPTYLATENTX_CLASS_MAPPINGS, EMPTYLATENTX_NAME_MAPPINGS
from .RemoveBackgroundX  import RMBG_CLASS_MAPPINGS, RMBG_NAME_MAPPINGS

WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_CLASS_MAPPINGS.update(GRADE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(GRADE_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(RELIGHT_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(RELIGHT_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(TILE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(TILE_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(DEPTHDISPLACE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(DEPTHDISPLACE_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(NF4BNB_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(NF4BNB_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(ACECOLORFIXX_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(ACECOLORFIXX_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(KSAMPLERCOMBOX_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(KSAMPLERCOMBOX_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(LORAX_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(LORAX_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(EMPTYLATENTX_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(EMPTYLATENTX_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(RMBG_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(RMBG_NAME_MAPPINGS)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', "WEB_DIRECTORY"]
