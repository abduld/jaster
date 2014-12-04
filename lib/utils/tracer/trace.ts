/// <reference path="count.ts" />

module lib.utils {
    export module trace {
        export module tracer {
            import count = lib.utils.trace.counter;

            interface LineLocation {
                start: Position
                end: Position
            }
            interface Position {
                line: number
                column: number
            }
            interface LogEvent {
            }
            interface NodeId {
                id: number;
            }
            class TracedNode {
                private log_:LogEvent[];
                private id_:NodeId;
                private loc_:LineLocation;

                constructor(loc:LineLocation) {

                }

                enter() {

                }

                exit() {

                }
            }
            class TracedFunction {
                private name_:string;
                private log_:LogEvent[];
                private args_:any[];
                private id_:NodeId;
                private fun_:Function
                constructor(fun:Function) {
                    this.fun_ = function (...args:any[]) {

                        var res = fun(args);

                        return res;
                    }
                }
            }
        }
    }
}
