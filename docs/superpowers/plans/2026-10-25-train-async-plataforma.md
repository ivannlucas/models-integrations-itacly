# Reentrenamiento asíncrono — plataforma · Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** que el seguimiento de un entrenamiento en la plataforma no corte a los 30 min, sobreviva a un reinicio sin relanzar el entrenamiento y avise por correo al usuario cuando acabe, bien o mal.

**Architecture:** decisión del usuario (2026-10-25): **reanudar la tarea**, no un barrido periódico. El proceso de la tarea `train` sigue consultando al orquestador, ahora sin límite de tiempo y tolerando caídas pasajeras. Si la plataforma se reinicia, `reassignUncompletedTasks` devuelve la tarea a `pending`; al relanzarse, `TrainTaskManager` ve el `orch_task_id` que ya guardó en `task.result` y retoma el seguimiento en vez de crear otro run y otro entrenamiento. La lógica nueva va en funciones puras (`src/service/train-tracking.ts`), que se pueden probar sin base de datos. `TrainTaskManager` solo las cablea.

**Tech Stack:** Node 20, TypeScript (`tsc` → `dist/`), Express, nodemailer, mocha sobre `dist/` (`npx tsc && npx mocha --recursive`).

**Spec:** `docs/superpowers/specs/2026-10-09-reentrenamiento-asincrono.md` (en `models-integrations-itacly`), punto 3 del diseño, con la decisión anterior.

**Repo de trabajo:** `C:\Users\PC\Documents\retech-lote2-xai-plataforma`, rama nueva `feature/async-train` desde `develop` (`5c18d38`), en la carpeta principal y sin worktrees. **Los commits los hace el usuario.**

## Global Constraints

- El contrato con el orquestador no cambia: `POST /api/ml/tasks/` y `GET /api/ml/tasks/{id}`, con los estados `queued | running | success | failed`.
- Sin límite fijo de tiempo de seguimiento: un entrenamiento perdido lo decide el servicio de modelos (`TrainJobLost`), y el orquestador lo devuelve como `failed`.
- Consultar cada `TRAIN_POLL_INTERVAL_MS = 10000`. Dar el seguimiento por fallido tras `TRAIN_POLL_MAX_CONSECUTIVE_ERRORS = 60` consultas seguidas sin respuesta válida (unos 10 min): errores de red, respuestas no 2xx o 404 del orquestador.
- Correo al usuario que lanzó la tarea (`task.user` → `User.email`), solo para entrenamientos que llegaron al orquestador. Va en español, como el resto de mensajes de entrenamiento. Un fallo al enviarlo nunca cambia el resultado de la tarea.
- En la reanudación no se repiten las comprobaciones de cuota, la preparación de datos, la creación del run de MLflow ni el envío al orquestador.
- Mensajes de error para el usuario: los que ya calcula `failTask` (`task.errorDescription`).
- Tests: `npx tsc --pretty false && npx mocha --recursive`. Línea base: 467 en verde.

## Review Focus

1. **Reinicio con la tarea ya terminada en el orquestador:** al reanudar, la primera consulta ya da `success`: se cierra sin volver a entrenar y se envía un solo correo. Test en Task 1 (`getResumableOrchTaskId`) y en Task 2 (primera consulta terminal).
2. **Orquestador reiniciándose** (conexión rechazada o 5xx durante unos minutos): el seguimiento sigue y el contador se reinicia con cada respuesta válida. Test en Task 2.
3. **Usuario sin email o con un email inválido, o SMTP caído:** no se envía nada, se registra en el log y la tarea mantiene su estado. Test en Task 3.
4. **Tarea reasignada antes de llegar al orquestador** (`task.result` vacío o sin `orch_task_id`): se ejecuta entera, como hoy. Test en Task 1.
5. **Tiempo de cómputo de la organización:** la tarea sigue contando mientras espera, como hoy. Con `tasksTimeLimit` puede acabar en `ERROR_TASK_TIMEOUT`. Se acepta y se documenta; sin test.

---

### Task 1: `getResumableOrchTaskId`

**Files:**
- Create: `src/service/train-tracking.ts`
- Test: `test/train-tracking.js` (nuevo; `require("../dist/service/train-tracking")`)

**Interfaces:**
- Produces: `export function getResumableOrchTaskId(task: { resultType?: string; result?: string }): string | null`. Devuelve el `orch_task_id` que `enqueueOrchTask` deja en `task.result` (`{"orch_task_id": "<id>"}` con `resultType` `application/json`); `null` en cualquier otro caso.

- [ ] **Step 1: Tests:**
  - `{"orch_task_id":"abc"}` → `"abc"`;
  - `result` vacío, `undefined` o `null` → `null`;
  - JSON roto → `null`;
  - `{"orch_task_id": ""}` → `null`;
  - `{"orch_task_id": 5}` → `null`;
  - `resultType: "model"` → `null`;
  - un resultado final (`{"orch_task_id":"abc","status":"failed"}`, el que escribe `failTask`) → `null`.
- [ ] **Step 2:** `npx tsc --pretty false && npx mocha test/train-tracking.js` → falla al resolver el módulo.
- [ ] **Step 3:** implementar.
- [ ] **Step 4:** el mismo comando → verde.

### Task 2: `pollOrchTaskUntilDone`

**Files:**
- Modify: `src/service/train-tracking.ts`
- Test: `test/train-tracking.js`

**Interfaces:**
- Produces:
  ```ts
  export const TRAIN_POLL_INTERVAL_MS = 10000;
  export const TRAIN_POLL_MAX_CONSECUTIVE_ERRORS = 60;
  export type OrchStatusResponse = { ok: boolean; status: number; data: any };
  export type PollOutcome =
      | { kind: "done"; data: any }                  // estado distinto de queued/running
      | { kind: "unreachable"; errors: number; last: OrchStatusResponse | null; error?: string };
  export async function pollOrchTaskUntilDone(deps: {
      fetchStatus: () => Promise<OrchStatusResponse>;  // puede lanzar (red)
      onProgress: (status: string) => Promise<void>;   // "QUEUED" | "RUNNING"
      sleep: (ms: number) => Promise<void>;
      intervalMs?: number;
      maxConsecutiveErrors?: number;
  }): Promise<PollOutcome>
  ```
  Bucle: `fetchStatus()`. Si lanza o `!ok`, cuenta un error, y al llegar a `maxConsecutiveErrors` devuelve `unreachable`. Si `status` (en minúsculas) es `queued` o `running`, pone el contador a 0 y llama a `onProgress(status.toUpperCase())`. Si es otro, devuelve `done`. Entre vueltas, `sleep(intervalMs)`.

- [ ] **Step 1: Tests**, con falsos que recorren una cola de respuestas:
  - `running`, `running`, `success` → `done` con los datos de `success`; `onProgress` recibe `["RUNNING", "RUNNING"]`; `sleep` se llama 2 veces con 10000.
  - primera respuesta `failed` → `done` sin `onProgress`.
  - `throw`, 503, `running`, `throw`, `success`, con `maxConsecutiveErrors: 3` → `done`.
  - siempre 502, con `maxConsecutiveErrors: 3` → `unreachable`, `errors === 3`, `last.status === 502`.
  - siempre `throw new Error("ECONNREFUSED")` → `unreachable` y `error` contiene `ECONNREFUSED`.
- [ ] **Step 2–4:** rojo → implementar → verde.

### Task 3: correo de fin de entrenamiento

**Files:**
- Create: `src/views/train.finished.email.ts`
- Modify: `src/service/alert-mailer.ts`
- Test: `test/train-tracking.js`

**Interfaces:**
- Produces:
  ```ts
  // src/views/train.finished.email.ts
  export type TrainFinishedMail = { modelName: string; success: boolean; detail: string; link: string };
  export class TrainFinishedEmailView {
      static subject(m: TrainFinishedMail): string  // "Datagia - Entrenamiento completado: <modelo>" | "Datagia - Entrenamiento fallido: <modelo>"
      static render(m: TrainFinishedMail): string    // HTML; todo valor pasa por escapeHTML
      static renderText(m: TrainFinishedMail): string
  }
  // src/service/alert-mailer.ts
  static async sendTrainingFinishedMail(to: string, m: TrainFinishedMail): Promise<boolean>
  ```
  `sendTrainingFinishedMail` devuelve `false` sin enviar si `!validateEmail(to)`. Si `sendMail` lanza, devuelve `false` y lo registra con `Monitor.error`. Nunca lanza. El transporte es el `mailTransporter` del módulo, con `from` como en los demás correos.

- [ ] **Step 1: Tests de la vista:**
  - `success: true` → el asunto contiene "completado" y el nombre del modelo;
  - `success: false` → el asunto contiene "fallido" y el HTML contiene `detail`;
  - un `modelName` con `<script>` sale escapado en el HTML;
  - el `link` aparece en el HTML y en el texto.
  - El caso de email inválido se prueba con `AlertMailer.sendTrainingFinishedMail("no-es-un-email", …)` → `false` (no llega a abrir conexión SMTP). Si cargar `alert-mailer` en mocha exige configuración, este caso se cubre en Task 4 con el `TrainTaskManager` y se deja una línea `Ruling:`.
- [ ] **Step 2–4:** rojo → implementar → verde.

### Task 4: `TrainTaskManager` — reanudar, seguir sin límite y avisar

**Files:**
- Modify: `src/service/train-task-manager.ts`
- Modify: `README.md` (sección corta "Entrenamiento: seguimiento y aviso por correo")
- Test: `test/train-tracking.js`. Si `TrainTaskManager` no se puede instanciar en mocha sin base de datos, la verificación de esta tarea es `tsc` + la suite + una revisión del cableado. Se deja una línea `Ruling:`.

Cambios:
- `start()`: justo después de leer `cfg`, `const resumeOrchTaskId = getResumableOrchTaskId(this.task)`. Si existe:
  - `task.message = "Resuming training tracking in orchestrator..."`;
  - `updateModelResults({ status: "RUNNING", orch_task_id })`;
  - saltar cuotas, `prepareData`, MLflow y `enqueueOrchTask`, e ir a `pollUntilDone` y `handleFinalStatus`.
  - Logar con `Monitor.info` que se reanuda.
- `pollUntilDone`: usa `pollOrchTaskUntilDone`; `fetchStatus` hace el `fetch` de siempre y `onProgress` hace `updateModelResults({ status, orch_task_id })`. Se quita el `TIMEOUT_MS` de 30 min y su comentario. Con `unreachable`: `failTask("ORCH_TRAIN_POLLING_FAILED", …)` con el mismo `modelPatch` y `technicalDetail` que hoy (HTTP y cuerpo de la última respuesta, o el error de red).
- Aviso: `private async notifyTrainingFinished(success: boolean): Promise<void>`.
  - Se llama al final de `handleFinalStatus`, en éxito y en los dos caminos de fallo, y tras el `failTask` de `unreachable`.
  - Busca el usuario con `User.findUserByUID(this.task.user)` y el nombre del modelo con `Model.findByID(cfg.model_id, cfg.ws_id)?.name`, con `cfg.model_key` de respaldo.
  - `detail`: en éxito, "El modelo está listo para usarse."; en fallo, `this.task.errorDescription`.
  - `link`: `Config.getInstance().getAbsoluteURI("/o/" + org + "?ws=" + ws + "&s=task&i=" + task.id)`.
  - Envuelto en `try/catch` con `Monitor.exception`: nunca lanza.
- `ERROR_DESCRIPTIONS["ORCH_TRAIN_POLLING_FAILED"]` pasa a ser "Se perdió la comunicación con el orquestador durante el entrenamiento. Inténtalo de nuevo más tarde.". `ORCH_TRAIN_POLLING_TIMEOUT` se conserva para tareas antiguas.

- [ ] **Step 1:** implementar.
- [ ] **Step 2:** `npx tsc --pretty false && npx mocha --recursive` → verde (467 + los nuevos).
- [ ] **Step 3:** README: reanudación, sin límite, 60 errores seguidos y correo; nota sobre `tasksTimeLimit`.

---

## Ejecución (2026-10-25)

Hecho en `retech-lote2-xai-plataforma`, rama `feature/async-train`, sin commit. `tsc` y ESLint sin
errores; mocha 485 tests en verde (467 de base + 18 nuevos en `test/train-tracking.js`).

Añadido tras la revisión independiente de la rama:

- `trackThenNotify`: si algo falla después del entrenamiento, la tarea queda `ERROR_TRAIN_FORWARD` y el
  usuario recibe igualmente el correo.
- `isStillTaskOwner`: antes de cerrar la tarea se relee de la base de datos; si se reasignó a otro proceso
  (reinicio solo del gestor de tareas), la cierra y avisa solo su dueño actual.
- `isResumedTrainTask`: `assignPendingTasks` no vuelve a aplicar las cuotas del usuario a una tarea
  reanudada.
- `TaskManager.timeout()`: no pisa una tarea ya terminada y, en un entrenamiento que llegó al orquestador,
  envía el correo de fallo.
- No reanuda si muere solo el proceso de la tarea (riesgo de bucle); documentado en el README.
- `TrainTaskManager` y `TaskManager.timeout()` no tienen test unitario (necesitan base de datos): la
  lógica nueva está en funciones puras de `train-tracking.ts`; el cableado se verifica en INT.
