
/// <reference path="../ref.ts" />
module lib {
    export module parallel {
        export class WorkerPool {
            private num_workers: number;
            private workers: Array<ParallelWorker>;
            constructor(num_workers: number) {
                this.num_workers = num_workers;
                this.workers = new Array<ParallelWorker>(num_workers);
            }
        }
    }
}