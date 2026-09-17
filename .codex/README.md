# Codex en MMARV1

[AGENTS.md](../AGENTS.md) contiene el contrato; [PROJECT.md](PROJECT.md) conserva
las reglas y referencias técnicas. El motor común vive en
[el hub](../../.codex/WORKSPACE.md). No se duplican agentes, hooks ni scripts
del motor en cada proyecto.

El perfil hereda los ajustes del usuario y se edita directamente. El marcador
profile.json impide que codex_sync.py regenere la proyección antigua.
Claude y los hooks Git existentes conservan sus archivos. No se ha cambiado
la confianza de hooks o proyectos ni la configuración global de Codex.

La configuración anterior y AGENTS.md se conservaron antes de la migración en
git rev-parse --git-path constellation/profile-before-native-2026-09-17.
Esa copia es local; la versión anterior también permanece en la historia Git.
Restaurar algo exige revisar y seleccionar archivos sin sobrescribir trabajo posterior.
