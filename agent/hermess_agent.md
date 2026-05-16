flowchart LR
    U[用户任务 / 对话] --> PB[Prompt Builder / Agent Loop]

    MEM[MEMORY.md / USER.md] --> PB
    IDX[Skills Index<br/>name / description / tags] --> PB

    CMD[/skill-name 或 自动选择] --> SC[skill_commands.py]
    SC --> SK[SKILL.md]
    SK --> REF[references / templates / scripts / assets]
    REF --> SC
    SC --> PB

    PB --> EXEC[LLM推理与工具执行]

    EXEC --> SM[skill_manage]
    SM --> A1[_create_skill]
    SM --> A2[_patch_skill]
    SM --> A3[_edit_skill]
    SM --> A4[_write_file]

    A1 --> VAL[结构校验<br/>name / frontmatter / size]
    A2 --> FUZZY[fuzzy_find_and_replace]
    A3 --> VAL
    A4 --> PATH[路径白名单 / 大小限制]

    FUZZY --> GUARD[安全扫描 / 回滚]
    VAL --> GUARD
    PATH --> GUARD

    GUARD --> STORE[~/.hermes/skills/]
    STORE --> IDX
    STORE --> SC

    FUTURE[下一步演进] --> F1[Patch历史 / 审计]
    FUTURE --> F2[质量校验 / lint]
    FUTURE --> F3[When to Use 索引]
    FUTURE --> F4[Per-skill 模型切换]
    FUTURE --> F5[离线自进化流水线]