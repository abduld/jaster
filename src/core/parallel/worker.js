/// <reference path="../../ref.ts" />
define(["require", "exports", "../../utils/utils"], function (require, exports, utils) {
    var Parallel;
    (function (Parallel) {
        var WorkerStatus;
        (function (WorkerStatus) {
            WorkerStatus[WorkerStatus["Paused"] = 0] = "Paused";
            WorkerStatus[WorkerStatus["Idle"] = 1] = "Idle";
            WorkerStatus[WorkerStatus["Busy"] = 2] = "Busy";
            WorkerStatus[WorkerStatus["Cancel"] = 3] = "Cancel";
        })(WorkerStatus || (WorkerStatus = {}));
        ;
        var INIT_PAUSE_LENGTH = 100; // milliseconds;
        var SharedWorker = (function () {
            function SharedWorker(fun, port) {
                this.timeout_handle = -1;
                this.id = utils.guuid();
                this.status = 1 /* Idle */;
                this.master_port = port;
                this.chan = new MessageChannel();
                // Build a worker from an anonymous function body
                var blobURL = URL.createObjectURL(new Blob(['(', fun.toString(), ')()'], { type: 'application/javascript' }));
                this.worker = new Worker(blobURL);
                // Won't be needing this anymore
                URL.revokeObjectURL(blobURL);
            }
            SharedWorker.prototype.run0 = function (init, end, inc) {
                var iter = init;
                if (this.status === 0 /* Paused */) {
                    this.pause_length *= 2;
                    setTimeout(this.run0, this.pause_length, [init, end, inc]);
                    return false;
                }
                if (this.timeout_handle !== -1) {
                    clearTimeout(this.timeout_handle);
                }
                while (iter < end) {
                    this.fun(iter);
                    if (this.status === 3 /* Cancel */) {
                        break;
                    }
                    else if (this.status === 0 /* Paused */) {
                        setTimeout(this.run0, this.pause_length, [iter + inc, end, inc]);
                        return false;
                    }
                    iter += inc;
                }
                this.status = 1 /* Idle */;
            };
            SharedWorker.prototype.run = function (fun, start_idx, end_idx, inc) {
                this.fun = fun;
                this.pause_length = INIT_PAUSE_LENGTH;
                this.status = 2 /* Busy */;
                if (inc) {
                    return this.run0(start_idx, end_idx, inc);
                }
                else {
                    return this.run0(start_idx, end_idx, 1);
                }
            };
            SharedWorker.prototype.cancel = function () {
                this.status = 3 /* Cancel */;
            };
            SharedWorker.prototype.pause = function () {
                this.status = 0 /* Paused */;
            };
            return SharedWorker;
        })();
        Parallel.SharedWorker = SharedWorker;
    })(Parallel || (Parallel = {}));
});
//# sourceMappingURL=worker.js.map