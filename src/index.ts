/// <reference path="utils.ts" />
/// <reference path="reactutils.ts" />
/// <reference path="visualization.ts" />
/// <reference path="memory.ts" />

import React = require("react");
import Visualization = require("visualization"); 

export var main = () => {
    var hostMemoryManager = new Core.HostMemoryManager();
    var mem = hostMemoryManager.malloc(20);
    var dom = document.getElementById("visualization");
    var rect = Visualization.gridVisualization({
        blockDim: new Core.Dim3(4, 4),
        gridDim: new Core.Dim3(16, 16)
    });
    React.render(rect, dom);
};
