# AGENTS.md — MMARV1

Biblioteca científica de análisis multifractal: estimadores, modelos de cascadas, inferencia con sustitutos y procedencia reproducible de resultados. El código legacy conserva la referencia histórica.

## Trabajo con Codex

Este es el contrato nativo del proyecto. Completa la tarea, verifica el resultado
y conserva cambios ajenos. Hereda el modelo, esfuerzo y permisos del usuario.
Consulta el [contexto técnico](.codex/PROJECT.md), especialmente sus invariantes,
antes de modificar comportamiento; verifica los comandos contra los archivos
actuales. Lee también los AGENTS.md de las carpetas afectadas.

Delega tareas independientes con archivos definidos; quien coordina integra,
revisa y publica. Hereda la elección de modelo del usuario; evita commits
simultáneos sobre el mismo repositorio.

## Repositorio y publicación

En la constelación, lee el [contrato común](../AGENTS.md) y usa el
[flujo entre repositorios](../.codex/WORKSPACE.md). Desde GitHub o desde aquí:

~~~sh
python3 ~/GitHub/.codex/workspace.py route MMARV1/AGENTS.md
~~~

Las rutas del coordinador son relativas a GitHub, independientemente del cwd.
Declara archivos exactos, comprueba el upstream antes de escribir, ejecuta las
pruebas pertinentes, revisa el diff y publica cada repositorio en su propio
upstream. Comprueba el SHA remoto. Nunca añadas este repositorio al índice
del padre ni empujes su trabajo desde otro repositorio. Publica los proveedores
antes que los consumidores cuando haya dependencias entre cambios.

En un clon sin el hub, realiza esos pasos con git -C <raíz_real> y las
herramientas del proyecto. No instales el hub incidentalmente. Los hooks Git
existentes siguen vigentes. No uses --no-verify, force-push, rebase, amend,
reset duro, clean o descarte de cambios sin autorización específica.
No inspecciones ni gestiones recuentos de worktrees.

Incluye solo archivos propios del encargo. Conserva un índice ajeno y resuelve
la autoría antes de publicar. No publiques prompts, transcripciones, archivos
de conversaciones, sus manifests, estados de sincronización, logs de daemons
ni secretos. No actives telemetría o Memories ni envíes mensajes sin permiso.

El marcador .codex/profile.json protege el perfil frente a la proyección.
Se edita directamente. La confianza de Codex se revisa en su interfaz;
nunca se modifican sus hashes.
