import React = require("react");
//import Utils = require("./utils");
import Visualization = require("./visualization");
import Memory = require("./memory");

export var main = () => {
    var hostMemoryManager = new Memory.HostMemoryManager();
    var mem = hostMemoryManager.malloc(20);
    console.log(Visualization.cell);
    console.log(mem);
    var dom = document.getElementById("content");
    var rect = React.DOM.svg(null, React.DOM.rect({width: 100, height: 100, fill: "black"}));
    console.log(React.render(rect, dom));
    console.log(React.renderToString(rect));
};

