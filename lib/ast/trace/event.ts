module lib.ast {
    export module trace {
        export class LogEvent {
            description: string;
            time: number;

            constructor(description: string) {
                this.description = description;
                this.time = lib.utils.timer.now();
            }
        }
    }
}
