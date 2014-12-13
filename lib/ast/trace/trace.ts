/// <reference path="count.ts" />
/// <reference path="event.ts" />


// see http://groups.csail.mit.edu/uid/other-pubs/tl-ms-thesis.pdf
module lib.ast {
    export module trace {

        interface LineLocation {
            start: Position
            end: Position
        }
        interface Position {
            line: number
            column: number
        }
        interface NodeId {
            id: number
        }
        class TracedNode {
            private log_: LogEvent[];
            private id_: NodeId;
            private loc_: LineLocation;

            constructor(loc: LineLocation) {
                this.loc_ = loc;
            }

            enter() {

            }

            exit() {

            }

            eval() {

            }
        }
        class TracedFunction extends TracedNode {
            private fun_: Function;

            constructor(loc: LineLocation, fun: Function) {
                var self = this;
                super(loc);
                this.fun_ = function(...args: any[]) {
                    self.enter();
                    var res = fun(args);
                    self.exit();
                    return res;
                }
            }

            eval() {

            }
        }

        export function makeTraceable(o: any) {

        }
    }
}
