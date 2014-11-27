

module lib.utils {
    export module parallel {
        export class Semaphore {
            private _curr : number;
            private _queue : Function[]= [];
            constructor(private n : number) {
                this._curr = n;
            }
            enter(fn:()=>void) {
                if (this._curr > 0) {
                    --this._curr;
                } else {
                    this._queue.push(fn);
                }
            }
            leave() {
                if (this._queue.length > 0) {
                    var fn = this._queue.pop();
                    fn();
                } else {
                    ++this._curr;
                }
            }
        }
    }
}