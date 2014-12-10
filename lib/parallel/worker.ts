

module lib.parallel {

export interface WebWorkerMessage {
    command : string;
    data?: any;
}

        export class WebWorker {
            script : string;
            id : number;
            private pool_ : typeof WorkerPool
            private worker_ : Worker

            constructor(id : number, script : string, pool : typeof WorkerPool) {
                this.id = id;
                this.script = script;
                this.pool_ = pool;
                this.worker_ = new Worker(script);
            }

            start() {
                this.worker_.postMessage({
                    command: "start"
                })
            }

            stop() {
                this.worker_.postMessage({
                    command: "stop"
                })
            }

            pause() {
                this.worker_.postMessage({
                    command: "pause"
                })
            }

            status() {
                this.worker_.postMessage({
                    command: "status"
                })
            }
        }
}