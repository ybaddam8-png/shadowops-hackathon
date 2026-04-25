export interface Scenario {
  id: string
  title: string
  domain: string
  intent: string
  payload: string
  riskLevel: 'critical' | 'high' | 'medium' | 'low'
  decision: 'ALLOW' | 'BLOCK' | 'QUARANTINE' | 'FORK'
  confidence: number
  reason: string
  riskVectors: string[]
  mireTactic: string
  mireTechnique: string
  timestamp: string
  evidence?: {
    status: 'detecting' | 'analyzing' | 'verified'
    result: string
    finalDecision?: 'ALLOW' | 'BLOCK' | 'QUARANTINE' | 'FORK'
    restrictions?: string
  }
}

export const SCENARIOS: Scenario[] = [
  {
    id: 'github-poisoned-pr',
    title: 'GitHub poisoned PR',
    domain: 'github',
    intent: 'Merge dependency update into production branch',
    payload: 'Adds post-install script and modifies CI secrets access',
    riskLevel: 'critical',
    decision: 'QUARANTINE',
    confidence: 82,
    reason: 'The request modifies CI execution and touches secret access. ShadowOps is holding the action until maintainer identity and diff evidence are verified.',
    riskVectors: [
      'Secret access requested',
      'CI workflow modified',
      'Unknown package maintainer',
      'Production branch target',
    ],
    mireTactic: 'Initial Access',
    mireTechnique: 'Supply Chain Compromise',
    timestamp: '2024-01-15T14:32:00Z',
    evidence: {
      status: 'verified',
      result: 'Maintainer identity confirmed with low fork risk',
      finalDecision: 'FORK',
      restrictions: 'Require code review and diff inspection',
    },
  },
  {
    id: 'aws-s3-exfil',
    title: 'AWS public S3 exfiltration',
    domain: 'aws',
    intent: 'Make S3 bucket publicly readable',
    payload: 'Set bucket policy to public access',
    riskLevel: 'critical',
    decision: 'BLOCK',
    confidence: 95,
    reason: 'Unrestricted public access to production data bucket violates security policy.',
    riskVectors: [
      'Public access flag',
      'Data exfiltration vector',
      'No authentication required',
      'Production data at risk',
    ],
    mireTactic: 'Exfiltration',
    mireTechnique: 'Exfiltration Over Unencrypted/Obfuscated Non-C2 Channel',
    timestamp: '2024-01-15T13:15:00Z',
  },
  {
    id: 'rogue-iam-admin',
    title: 'Rogue IAM admin creation',
    domain: 'iam',
    intent: 'Create new admin user account',
    payload: 'Add user with AdministratorAccess policy',
    riskLevel: 'critical',
    decision: 'QUARANTINE',
    confidence: 88,
    reason: 'Privilege escalation attempt detected. Requires maintainer verification and audit log review.',
    riskVectors: [
      'Admin privilege grant',
      'New principal created',
      'No approval chain detected',
      'Unusual timing',
    ],
    mireTactic: 'Persistence',
    mireTechnique: 'Create Account',
    timestamp: '2024-01-15T12:45:00Z',
  },
  {
    id: 'firewall-rule-open',
    title: 'Suspicious firewall opening',
    domain: 'network',
    intent: 'Open ingress rule on port 3389',
    payload: 'Allow RDP from 0.0.0.0/0',
    riskLevel: 'high',
    decision: 'QUARANTINE',
    confidence: 76,
    reason: 'Unrestricted RDP access creates lateral movement risk. Needs context on maintenance window.',
    riskVectors: [
      'Unrestricted CIDR range',
      'RDP exposure',
      'No time restriction',
      'Production environment',
    ],
    mireTactic: 'Lateral Movement',
    mireTechnique: 'Remote Service Session Hijacking',
    timestamp: '2024-01-15T11:20:00Z',
  },
  {
    id: 'pentest-request',
    title: 'Ambiguous pentest request',
    domain: 'pentest',
    intent: 'Execute security assessment actions',
    payload: 'Run vulnerability scan and privilege escalation tests',
    riskLevel: 'medium',
    decision: 'QUARANTINE',
    confidence: 72,
    reason: 'Authorized test window could not be confirmed. Awaiting pentest coordinator approval.',
    riskVectors: [
      'External security test',
      'Privilege escalation tools',
      'Scope ambiguity',
      'No explicit authorization',
    ],
    mireTactic: 'Discovery',
    mireTechnique: 'Account Discovery',
    timestamp: '2024-01-15T10:00:00Z',
    evidence: {
      status: 'verified',
      result: 'Authorized test window confirmed',
      finalDecision: 'ALLOW',
      restrictions: 'Restrict to internal CIDR, log all actions',
    },
  },
]

export const DECISION_TYPES = [
  {
    label: 'ALLOW',
    color: 'bg-green-900 text-green-300 border-green-700',
    icon: '✓',
    description: 'Action authorized and safe',
  },
  {
    label: 'BLOCK',
    color: 'bg-red-900 text-red-300 border-red-700',
    icon: '✕',
    description: 'Action blocked - security policy violation',
  },
  {
    label: 'QUARANTINE',
    color: 'bg-amber-900 text-amber-300 border-amber-700',
    icon: '⏸',
    description: 'Action held pending evidence',
  },
  {
    label: 'FORK',
    color: 'bg-purple-900 text-purple-300 border-purple-700',
    icon: '→',
    description: 'Escalated to human review',
  },
]
