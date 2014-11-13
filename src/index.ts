import React = require("react");
import Utils = require("./utils");
import Visualization = require("./visualization");
import Memory = require("./memory");

export var main = () => {
    var hostMemoryManager = new Memory.HostMemoryManager();
    var mem = hostMemoryManager.malloc(20);
    var dom = document.getElementById("visualization");
    var rect = Visualization.gridVisualization({
        blockDim: new Utils.Dim3(4, 4),
        gridDim: new Utils.Dim3(16, 16)
    });
    React.render(rect, dom);
};
