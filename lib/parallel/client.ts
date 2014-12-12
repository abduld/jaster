module lib {
    export module parallel {

        var port:MessagePort = undefined;
        var id:string;

        export function actOnEvent(event:MessageEvent) {
            switch (event.data.command) {
                case "setPort":
                    port = event.ports[0];
                    break ;
                case "setConsole":
                    if (_.isUndefined(global$.console._port)) {
                        global$.console = {
                            _port: port,           // Remember the port we log to
                            log: function log() {        // Define console.log()
                                // Copy the arguments into a real array
                                var args = Array.prototype.slice.call(arguments);
                                // Send the arguments as a message, over our side channel
                                port.postMessage({
                                    command: "log",
                                    id: id,
                                    data: args
                                });
                            }
                        };
                    }
                    console.log("got command " + event.data.command);
                    break;
                case "setId":
                    id = event.data.data;
                    break;
                default:
                    console.log("got event " + JSON.stringify(event.data));
            }
        }
    }
}

