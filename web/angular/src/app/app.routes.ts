import { Routes } from '@angular/router';
import { AgUiPageComponent } from './pages/ag-ui-page.component';
import { AgenticResearchPageComponent } from './pages/agentic-research-page.component';
import { LandingPageComponent } from './pages/landing-page.component';
import { TrainingPageComponent } from './pages/training-page.component';

export const routes: Routes = [
  { path: '', component: LandingPageComponent },
  { path: 'agentic-research', component: AgenticResearchPageComponent },
  { path: 'ml/pytorch', component: TrainingPageComponent, data: { framework: 'pytorch' } },
  { path: 'ml/tensorflow', component: TrainingPageComponent, data: { framework: 'tensorflow' } },
  { path: 'ag-ui', component: AgUiPageComponent },
  { path: '**', redirectTo: '' }
];
