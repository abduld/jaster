import React = require("react");
import Utils = require("./utils");
import Visualization = require("./visualization");
import Memory = require("./memory");

export var main = () => {
    var hostMemoryManager = new Memory.HostMemoryManager();
    var mem = hostMemoryManager.malloc(20);
    console.log(Visualization.cell);
    console.log(mem);
    var dom = document.getElementById("content");
    var rect = Visualization.gridVisualization({
        blockDim: new Utils.Dim3(2, 2),
        gridDim: new Utils.Dim3(2, 2)
    });
    console.log(React.render(rect, dom));
    console.log(React.renderToString(rect));
};
