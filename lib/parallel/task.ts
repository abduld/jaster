

module lib {
    export module parallel {
        class Task {
            constructor() {

            }
        }

        export class TaskScheduler {
            tasks: Task[];
            constructor() {

            }
            addTask(f: Function) {
                tasks.push(f);
            }

        }
    }
}