
/// <reference path="ref.ts" />
module Core {

    export var guuid = () : string => {
        var s4 = () : string =>
            Math.floor((1 + Math.random()) * 0x10000)
                .toString(16)
                .substring(1);
        return s4() + s4() + "-" + s4() + "-" + s4() + "-" +
            s4() + "-" + s4() + s4() + s4();
    };

    export class Dim3 {
        public x : number;
        public y : number;
        public z : number;
        constructor(x : number, y = 1, z = 1) {
            this.x = x;
            this.y = y;
            this.z = z;
        }
    }

    export var rand = function (min:number, max:number):number {
        return min + Math.random() * (max - min);
    }
    export function applyMixins(derivedCtor: any, baseCtors: any[]) {
        baseCtors.forEach(baseCtor => {
            Object.getOwnPropertyNames(baseCtor.prototype).forEach(name => {
                derivedCtor.prototype[name] = baseCtor.prototype[name];
            })
        });
    }
}
export = Core;