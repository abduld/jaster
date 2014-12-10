interface Navigator {
    hardwareConcurrency : number;
}
interface Document {
    currentScript : any;
}


/// <reference path="worker.ts" />
module lib {
    export module parallel {
        class WorkerPool_ {
            private workers_:WebWorker[];
            private size:number;
            private pending_:Function[];
            private retired_:Function[];
            private mask_:boolean[]

            constructor(size:number) {
                this.size = size;
                this.workers_ = _.map(_.range(size), (idx) => new WebWorker(idx, document.currentScript.src, this));
            }

            private recieveMessage_(idx:number, msg : MessageEvent) {

            }

            private recieveError_(idx:number, msg : MessageEvent) {

            }

            start() {
                _.each(this.workers_,
                    (worker:WebWorker) => worker.start()
                );
            }

            pause() {

            }

            resume() {

            }

            cancel() {

            }
        }
        export var WorkerPool:WorkerPool_ = null;
        if (lib.utils.ENVIRONMENT_IS_WEB) {
            WorkerPool = new WorkerPool_(navigator.hardwareConcurrency || 4);
        }
    }
}