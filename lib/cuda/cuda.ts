
module lib {
    export module cuda {
        export enum Status {
            Running,
            Idle,
            Complete,
            Stopped,
            Waiting
        }
        export class Dim3 {
            x : number;
            y : number;
            z : number;
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
    }
}
