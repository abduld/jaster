

module lib {
    export function init() {
        return {
            type: "GlobalState",
            id: lib.utils.guuid()
        }
    }

    export function chceckEvent(state, worker, functionStack) {
        return false;
    }
    export function handleEvent(state, worker, functionStack) {
        return false;
    }
}