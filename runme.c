/* Modified from PETSc TS tutorial ex3 - see there for more options/analysis/comments on this system */
/* See things marked NEW: */
static char help[] = "Demonstrate generic TSRollBack() implementation.\n\
Solves a simple time-dependent linear PDE (the heat equation).\n\
Input parameters include:\n\
  -m <points>, where <points> = number of grid points\n\n";

#include <petscts.h>
#include <petscdraw.h>

/* NEW: include to allow non-API operations in TSRollBackGenericActivate() */
#include <petsc/private/tsimpl.h>

typedef struct {
  PetscInt    m;
  PetscReal   h;
  Mat         A;
} AppCtx;

extern PetscErrorCode InitialConditions(Vec,AppCtx*);
extern PetscErrorCode RHSMatrixHeat(TS,PetscReal,Vec,Mat,Mat,void*);

/* NEW: generic rollback impl */
PetscErrorCode TSRollBack_Generic(TS ts)
{
  PetscErrorCode ierr;
  Vec            X,Xprev;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)ts,"RollBackGeneric_Xprev",(PetscObject*)(&Xprev));CHKERRQ(ierr);
  if (!Xprev) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_WRONGSTATE,"Rollback data not stored before TSRollBack() called (no steps taken?)");
  ierr = TSGetSolution(ts,&X);CHKERRQ(ierr);
  ierr = VecCopy(Xprev,X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* NEW: prestep function to save state. With a flag check,
   this could be included in the TS implementation */
PetscErrorCode PreStep_RollBackGeneric(TS ts)
{
  PetscErrorCode ierr;
  Vec            X,Xprev;

  PetscFunctionBegin;
  ierr = TSGetSolution(ts,&X);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)ts,"RollBackGeneric_Xprev",(PetscObject*)(&Xprev));CHKERRQ(ierr);
  if (!Xprev) {
    ierr = VecDuplicate(X,&Xprev);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)ts,"RollBackGeneric_Xprev",(PetscObject)Xprev);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)Xprev);CHKERRQ(ierr);
  }
  ierr = VecCopy(X,Xprev);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* NEW: activate new impl. Note that this uses the ts->prestep slot,
        but could be integrated into the TS implementation with dedicated logic */
PetscErrorCode TSRollBackGenericActivate(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ts->setupcalled) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_WRONGSTATE,"Cannot activate generic rollback before calling TSSetUp()");
  if (ts->ops->rollback) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_WRONGSTATE,"Refusing to activate generic rollback for a TS implementation that already implements TSRollBack()");
  ts->ops->rollback = TSRollBack_Generic;
  ierr = TSSetPreStep(ts,PreStep_RollBackGeneric);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* NEW: example of user post-step hook to roll back and stop */
PetscErrorCode PostStep_User(TS ts)
{
  PetscErrorCode ierr;
  PetscInt       stepNumber;

  PetscFunctionBeginUser;
  ierr = TSGetStepNumber(ts,&stepNumber);CHKERRQ(ierr);
  if (stepNumber > 4) {
    ierr = PetscPrintf(PetscObjectComm((PetscObject)ts),"Custom PostStep - rolling back and stopping after 4 steps.\n");
    ierr = TSRollBack(ts);CHKERRQ(ierr);
    ierr = TSSetConvergedReason(ts,TS_CONVERGED_USER);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  AppCtx         appctx;
  TS             ts;
  Mat            A;
  Vec            u;
  PetscReal      time_total_max = 100.0;
  PetscInt       time_steps_max = 100;
  PetscErrorCode ierr;
  PetscInt       m;
  PetscMPIInt    size;
  PetscReal      dt;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_SELF,1,"This is a uniprocessor example only!");

  m    = 60;
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);

  appctx.m        = m;
  appctx.h        = 1.0/(m-1.0);

  ierr = VecCreateSeq(PETSC_COMM_SELF,m,&u);CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_SELF,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_LINEAR);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_SELF,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,m);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  appctx.A = NULL;
  ierr = RHSMatrixHeat(ts,0.0,u,A,A,&appctx);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,TSComputeRHSFunctionLinear,&appctx);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,A,A,TSComputeRHSJacobianConstant,&appctx);CHKERRQ(ierr);

  dt   = appctx.h*appctx.h/2.0;
  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);

  ierr = TSSetMaxSteps(ts,time_steps_max);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,time_total_max);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = InitialConditions(u,&appctx);CHKERRQ(ierr);

  /* NEW: add generic implementation for rollback */
  ierr = TSSetSolution(ts,u);CHKERRQ(ierr); /* required for TSSetUp() */
  ierr = TSSetUp(ts);CHKERRQ(ierr);
  ierr = TSRollBackGenericActivate(ts);CHKERRQ(ierr);

  /* NEW: add poststep function to rollback and stop based on some criterion */
  ierr = TSSetPostStep(ts,PostStep_User);CHKERRQ(ierr);

  ierr = TSSolve(ts,NULL);CHKERRQ(ierr);

  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = MatDestroy(&appctx.A);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode InitialConditions(Vec u,AppCtx *appctx)
{
  PetscScalar    *u_localptr,h = appctx->h;
  PetscErrorCode ierr;
  PetscInt       i;

  ierr = VecGetArray(u,&u_localptr);CHKERRQ(ierr);
  for (i=0; i<appctx->m; i++) u_localptr[i] = PetscSinScalar(PETSC_PI*i*6.*h) + 3.*PetscSinScalar(PETSC_PI*i*2.*h);
  ierr = VecRestoreArray(u,&u_localptr);CHKERRQ(ierr);

  return 0;
}

PetscErrorCode RHSMatrixHeat(TS ts,PetscReal t,Vec X,Mat AA,Mat BB,void *ctx)
{
  Mat            A       = AA;
  AppCtx         *appctx = (AppCtx*)ctx;
  PetscInt       mstart  = 0;
  PetscInt       mend    = appctx->m;
  PetscErrorCode ierr;
  PetscInt       i,idx[3];
  PetscScalar    v[3],stwo = -2./(appctx->h*appctx->h),sone = -.5*stwo;

  mstart = 0;
  v[0]   = 1.0;
  ierr   = MatSetValues(A,1,&mstart,1,&mstart,v,INSERT_VALUES);CHKERRQ(ierr);
  mstart++;

  mend--;
  v[0] = 1.0;
  ierr = MatSetValues(A,1,&mend,1,&mend,v,INSERT_VALUES);CHKERRQ(ierr);

  v[0] = sone; v[1] = stwo; v[2] = sone;
  for (i=mstart; i<mend; i++) {
    idx[0] = i-1; idx[1] = i; idx[2] = i+1;
    ierr   = MatSetValues(A,1,&i,3,idx,v,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  return 0;
}
