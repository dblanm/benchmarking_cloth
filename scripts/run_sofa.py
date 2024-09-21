import SofaRuntime
from bcm.envs.sofa.cloth_dyn_sofa import ClothDynSofa


def createScene(rootNode):
    rootNode.findData("dt").value = 0.05
    ClothDynSofa(rootNode)
    return 0
