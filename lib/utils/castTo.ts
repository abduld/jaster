module lib {
    export module utils {
        export function castTo<T>(arg: any): T {
            return <T> arg;
        }
    }
}

