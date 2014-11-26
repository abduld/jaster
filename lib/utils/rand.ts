
module lib {
    export module utils {
        export function rand(min: number, max: number): number {
           return min + Math.random() * (max - min);
        }
    }
}
