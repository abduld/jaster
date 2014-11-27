
/// <reference path="../../ref.ts" />



module lib.cuda.exec {
    export class Warp {
        public id:string = utils.guuid();
        public thread:Thread = null;
    }
}
