# Changelog

所有重要的产品设计和需求变更都记录在此文件中。

## [0.2.0] - 2026-06-09

### Fixed
- **注册表装饰器**：`register_factor` 改为仅关键字参数，消除 `@register_factor("name")` 位置参数歧义
- **CLI 导入路径**：统一所有 `importlib.import_module()` 使用 `factor_pipeline.*` 全路径
- **注册表 clear 方法**：新增 `FactorRegistry.clear()`（之前测试中调用但未实现）
- **依赖声明**：补充 `akshare` 和 `baostock` 为 `[project.optional-dependencies].data`

### Changed
- **设计文档**：填充完整的架构设计、数据模型、API 定义、重构优先级
- **配置**：新增 `pip install -e ".[data]"` 数据采集可选依赖

### Status
✅ 已实现

## [0.1.0] - 2026-06-09

### Added
- 棕地导入：从现有项目创建设计文档，版本号 v0.1.0

### Status
✅ 已实现

---
