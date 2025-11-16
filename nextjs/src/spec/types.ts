export type ModuleId = "crypto" | "real-estate" | "job-assistant" | "events" | "documents" | "dashboard";

export type TestDiscipline = "sharp" | "suif";

export interface RouteSpec {
  path: string;
  title: string;
  description: string;
  pattern?: string;
  discipline: TestDiscipline;
}

export interface DataSourceSpec {
  id: string;
  name: string;
  description: string;
  docsUrl: string;
  type: "api" | "database" | "cache" | "auth";
  critical: boolean;
}

export interface TableSpec {
  name: string;
  description: string;
  columns: string[];
  tags: ("core" | "cache" | "embedding")[];
}

export interface AgentCapabilitySpec {
  id: string;
  description: string;
  workflow: string[];
}

export interface FeatureSpec {
  id: string;
  title: string;
  description: string;
  routes: string[];
  tests: Partial<Record<TestDiscipline, string[]>>;
}

export interface ModuleSpec {
  id: ModuleId;
  title: string;
  summary: string;
  status: "in-progress" | "planned" | "future";
  entryRoute: string;
  designPattern: "sharp" | "suif" | "hybrid";
  routes: RouteSpec[];
  dataSources: DataSourceSpec[];
  tables: TableSpec[];
  features: FeatureSpec[];
  agentCapabilities: AgentCapabilitySpec[];
}

export interface SharedComponentSpec {
  id: string;
  title: string;
  description: string;
}

export interface TestingSpec {
  sharpFocus: string[];
  suifFocus: string[];
  coverageGoals: string[];
  mocking: string[];
}

export interface ProjectSpec {
  name: string;
  description: string;
  stack: string[];
  patternStrategy: string;
  modules: ModuleSpec[];
  sharedComponents: SharedComponentSpec[];
  testing: TestingSpec;
  security: string[];
  monitoring: string[];
}
