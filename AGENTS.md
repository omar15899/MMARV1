# AGENTS.md — MMARV1

Biblioteca científica de análisis multifractal: estimadores, modelos de cascadas, inferencia con sustitutos y procedencia reproducible de resultados. El código legacy conserva la referencia histórica.

## Trabajo con Codex

Lee el [contexto técnico](.codex/PROJECT.md) y los AGENTS.md de las carpetas
afectadas; contrasta los comandos con los archivos actuales. El [perfil
nativo](.codex/README.md) se edita directamente: conserva `.codex/profile.json`.

Si este clon pertenece al hub y `../.codex/ecosystem/cli.py` está disponible,
aplica su [contrato](../AGENTS.md) y el [flujo común](../.codex/WORKSPACE.md),
con rutas relativas al hub y el subcomando `workspace`. En un clon independiente
usa `git -C <raíz_real>` y las herramientas del proyecto; no instales el hub
incidentalmente.

Sin hub, delimita archivos exactos y comprueba el upstream antes de editar.
Completa el encargo, ejecuta pruebas pertinentes y revisa el diff. Publica cada
repositorio en su propio upstream, nunca desde otro ni en el índice del padre;
verifica el SHA remoto y publica proveedores antes de consumidores. Delega
trabajo independiente con archivos definidos; quien coordina integra, revisa
y publica, sin commits simultáneos sobre el mismo repositorio.

En todo clon, hereda modelo, esfuerzo y permisos. Conserva cambios e índice
ajenos; publica sólo el alcance con autoría resuelta. Respeta los hooks: no uses
`--no-verify`, force-push, rebase, amend, reset duro, clean ni descartes sin
autorización específica. No inspecciones ni gestiones recuentos de worktrees.
Mantén locales prompts, transcripciones, conversaciones, sus manifests, estados
de sincronización y logs de daemons. No publiques secretos. No actives telemetría
o Memories ni envíes mensajes sin permiso. Revisa la confianza en Codex sin
alterar sus hashes.
