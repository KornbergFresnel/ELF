from .game_MC.game import Loader as mc_loader
from .game_CF.game import Loader as cf_loader
from .game_TD.game import Loader as td_loader

Loader = {
    "mc": mc_loader,
    "cf": cf_loader,
    "td": td_loader
}