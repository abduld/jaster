
/// <reference path="../../ref.ts" />

module lib.cuda.exec {

    export class Barrier {
        lineNumber: number;
        thread: Thread;
        constructor(lineNumber: number, thread: Thread) {
            this.lineNumber = lineNumber;
            this.thread = thread;
        }
    }

    export class BarrierGroup {
        // you can represent this as a typed UInt8Array,
        // but it does not make a difference since you'd
        // be optimizing for one browser (see https://github.com/sq/JSIL/issues/250)
        mask: Array<boolean>;
        constructor(dim: lib.cuda.Dim3) {
            this.mask = new Array(dim.flattenedLength());
        }
    }
}