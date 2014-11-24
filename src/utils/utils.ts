/// <reference path="../ref.ts" />
module Core {

    export enum ErrorCode {
        Success,
        MemoryOverflow,
        IntegerOverflow,
        Unknown
    };
    export class Error {
        code : ErrorCode;
        constructor(code ? : ErrorCode) {
            if (code) {
                this.code = code;
            } else {
                this.code = ErrorCode.Success;
            }
        }
    }

    export class Dim3 {
        public x : number;
        public y : number;
        public z : number;
        constructor(x : number, y = 1, z = 1) {
            this.x = x;
            this.y = y;
            this.z = z;
        }
        flattenedLength() : number {
            return this.x * this.y * this.z;
        }
        dimension() : number {
            if (this.z == 1) {
                if (this.y == 1) {
                    return 1;
                } else {
                    return 2;
                }
            } else {
                return 3;
            }
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