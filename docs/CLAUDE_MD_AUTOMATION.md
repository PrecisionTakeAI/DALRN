# CLAUDE.md Automation Documentation

## Overview

This repository includes an automated workflow that maintains the `CLAUDE.md` file - a comprehensive codebase reference document used by Claude Code to understand your project's architecture, patterns, and conventions.

## How It Works

### Automatic Triggers
- **On Code Changes**: Automatically runs when code files are pushed to main/master/develop branches
- **On PR Merge**: Triggers when pull requests are merged to update documentation
- **Weekly Schedule**: Runs every Sunday at 2 AM UTC as a safety net
- **Manual Dispatch**: Can be triggered on-demand via GitHub Actions UI

### Smart Triggering
- **Frequency Limit**: Won't run more than once every 4 hours to prevent spam
- **Change Detection**: Only runs if significant code changes are detected
- **Path Filtering**: Ignores documentation changes to avoid infinite loops
- **PR Merge Only**: For pull_request events, only runs on successful merge

### Process Flow
1. **Change Detection**: Compares codebase fingerprints to identify significant changes
2. **Context Gathering**: Analyzes repository structure, tech stack, and statistics
3. **Deep Analysis**: Claude examines 50+ representative files across the codebase
4. **Document Generation**: Creates/updates CLAUDE.md with current patterns and practices
5. **Validation**: Ensures all required sections are present and properly formatted
6. **PR Creation**: Opens a pull request for review (or saves results in dry-run mode)

## Configuration

### Using `.claude-config.yml`

Customize the automation behavior by creating a `.claude-config.yml` file in your repository root:

```yaml
analysis:
  max_files_to_analyze: 1000
  exclude_patterns:
    - "**/vendor/**"
    - "*.min.js"

claude_md:
  required_sections:
    - "PROJECT OVERVIEW"
    - "ARCHITECTURE"
    # Add your required sections

workflow:
  triggers:
    on_code_change: true  # Run when code is pushed
    min_hours_between_runs: 4  # Prevent runs within 4 hours
    min_change_percentage: 2  # Require 2% file changes
    on_pr_merge: true  # Run when PRs are merged
    
  create_draft_pr: true
  auto_reviewers:
    - "@username"
```

See the full configuration example in [.claude-config.yml](../.claude-config.yml).

### Workflow Dispatch Options

When manually triggering the workflow:

- **Analysis Depth**: 
  - `quick`: Basic pattern detection (<1000 files)
  - `standard`: Common patterns (<5000 files)
  - `comprehensive`: All files with detailed patterns (default)
  - `exhaustive`: Deep analysis with extensive examples

- **Focus Areas**: Comma-separated list (e.g., `security,testing,api`)
- **Dry Run**: Analyze without creating a PR
- **Force Update**: Update even if no changes detected
- **Notification Channel**: Where to send results (`none`, `slack`, `email`, `issue`)

## Setup Instructions

### Prerequisites

1. **Authentication** (choose one):
   - **API Key**: Add `ANTHROPIC_API_KEY` to repository secrets
   - **OAuth Token**: Generate with `claude setup-token` and add as `CLAUDE_CODE_OAUTH_TOKEN`

2. **GitHub App**: Install the [Claude GitHub App](https://github.com/apps/claude) on your repository

### Quick Start

1. The workflow file is already configured at `.github/workflows/claude-md-maintenance.yml`
2. Add your authentication token to repository secrets
3. (Optional) Customize settings in `.claude-config.yml`
4. Test with a manual run: Actions → Claude MD Maintenance → Run workflow

## Understanding the Output

### CLAUDE.md Structure

The generated file contains:

1. **Metadata Header**: Version, update date, analysis parameters
2. **Project Overview**: Purpose, problem solved, tech stack
3. **Architecture**: System design, data flow, deployment
4. **Core Components**: Main modules and their responsibilities
5. **API Endpoints**: Routes, auth, rate limiting (if applicable)
6. **Data Models**: Schemas, entities, relationships
7. **Key Algorithms**: Business logic, calculations, workflows
8. **Dependencies**: Libraries, versions, security notes
9. **Configuration**: Environment variables, settings
10. **Current State**: TODOs, technical debt, WIP features
11. **Entry Points**: How to start and explore the codebase

### Version Management

CLAUDE.md versions follow semantic versioning:
- **Major**: Significant architectural changes
- **Minor**: New sections or substantial updates
- **Patch**: Minor corrections and clarifications

## Troubleshooting

### Common Issues

#### Workflow Fails with "No CLAUDE.md created"
- Check Claude API limits and quotas
- Verify authentication tokens are valid
- Review workflow logs for specific errors

#### PR Contains Incorrect Information
- Adjust analysis depth for better coverage
- Add focus areas to target specific aspects
- Update `.claude-config.yml` exclusion patterns

#### Workflow Times Out
- Reduce `max_files_to_analyze` in configuration
- Use `standard` or `quick` analysis depth
- Exclude large generated directories

#### No Changes Detected (when changes exist)
- Use `force_update: true` in manual dispatch
- Clear workflow cache in Actions settings
- Check if changes are in excluded patterns

### Manual Override

To bypass automation and update manually:
1. Edit CLAUDE.md directly
2. Add `<!-- manual-section -->` comments around custom content
3. These sections will be preserved during automated updates

## Cost Estimation

Approximate token usage per run:

| Codebase Size | Analysis Depth | Estimated Tokens | Estimated Cost* |
|---------------|----------------|------------------|-----------------|
| Small (<1K files) | Quick | ~10,000 | ~$0.15 |
| Medium (<5K files) | Standard | ~30,000 | ~$0.45 |
| Large (<10K files) | Comprehensive | ~75,000 | ~$1.13 |
| Very Large (>10K files) | Exhaustive | ~150,000 | ~$2.25 |

*Based on Claude Opus pricing as of 2024. Actual costs may vary.

## Best Practices

### For Accuracy
1. **Run after major merges**: Trigger manually after significant changes
2. **Review PRs carefully**: Automated analysis may miss nuances
3. **Maintain manual sections**: Add project-specific context that automation can't capture

### For Efficiency
1. **Configure exclusions**: Skip generated/vendored code
2. **Use appropriate depth**: Match analysis to codebase size
3. **Enable caching**: Reduces redundant processing

### For Team Adoption
1. **Document custom conventions**: Add team-specific guidelines manually
2. **Set up notifications**: Keep team informed of updates
3. **Regular reviews**: Schedule quarterly manual reviews

## Security Considerations

### Sensitive Information
- The workflow automatically redacts common patterns (API keys, tokens)
- Review generated content for inadvertent exposure
- Use `security.skip_sensitive_files` in config

### Access Control
- Workflow requires write permissions for contents and pull-requests
- Consider using branch protection rules
- Limit who can trigger manual runs

## Advanced Features

### Parallel Analysis
For large monorepos, enable parallel processing:
```yaml
performance:
  parallel_processing: true
  max_workers: 8
```

### Custom Notifications
Configure Slack notifications:
```yaml
notifications:
  slack:
    enabled: true
    channel: "#dev-team"
```

### Compliance Checking
Validate licenses and security:
```yaml
security:
  compliance:
    check_licenses: true
    allowed_licenses: ["MIT", "Apache-2.0"]
```

## Monitoring and Metrics

### Workflow Metrics
- **Duration**: Typically 5-30 minutes depending on size
- **Success Rate**: Track in Actions → Claude MD Maintenance
- **Cache Hit Rate**: Visible in workflow logs

### Quality Metrics
Each run reports:
- Sections coverage
- File analysis count  
- Change detection accuracy
- Validation warnings

## Contributing

To improve the automation:
1. Test changes in a fork first
2. Use dry-run mode for validation
3. Document any new configuration options
4. Update this guide with lessons learned

## Support

- **Issues**: Report problems in GitHub Issues with `claude-md-automation` label
- **Discussions**: Share tips and tricks in GitHub Discussions
- **Updates**: Watch the repository for workflow improvements

---

*Last updated: 2024-01-16*
*Workflow version: 1.0.0*