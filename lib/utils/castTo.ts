

module lib {
    export module utils {
        export module detail {
            export function castTo<T>(arg: any): T {
                return <T> arg;
            }
        }
    }
}

