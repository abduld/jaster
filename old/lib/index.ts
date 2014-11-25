
/// <reference path="ref.ts" />


import React = require("react");
import Visualization = require("./viz/visualization");
import core = require("./core/core");
import utils = require("./utils/utils");
import memory = require("./core/mem/memory");

export var main = () => {
    var hostMemoryManager = new memory.HostMemoryManager();
    var mem = hostMemoryManager.malloc(20);
    var dom = document.getElementById("visualization");
    var i3 = new core.int8(2);
    var rect = Visualization.gridVisualization({
        blockDim: new utils.Dim3(4, 4),
        gridDim: new utils.Dim3(16, 16)
    });
    //React.render(rect, dom);
};
