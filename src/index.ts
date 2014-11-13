import Utils = require("./utils");
import Visualization = require("./visualization");
import Memory = require("./memory");

var main = () => {
    var hostMemoryManager = new Memory.HostMemoryManager();
    var mem = hostMemoryManager.malloc(20);
    console.log(mem);
    React.renderToStaticMarkup(Visualization.cell({threadGroup: Utils.guuid()}));
};

window.onload = () => {
    main();
};
