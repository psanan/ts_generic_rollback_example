/* Modified from PETSc TS tutorial ex3 - see there for more options/analysis/comments on this system */
/* See things marked NEW: */
static char help[] ="Solves a simple time-dependent linear PDE (the heat equation).\n\
Input parameters include:\n\
  -m <points>, where <points> = number of grid points\n\n";

#include <petscts.h>
#include <petscdraw.h>

/* NEW - allows invasive (non-API) operations in TSRollBackGenericActivate() */
#include <petsc/private/tsimpl.h>

typedef struct {
  Vec         solution;          /* global exact solution vector */
  PetscInt    m;                 /* total number of grid points */
  PetscReal   h;                 /* mesh width h = 1/(m-1) */
  Mat         A;                 /* RHS mat, used with IFunction interface */
  PetscReal   oshift;            /* old shift applied, prevent to recompute the IJacobian */
} AppCtx;

extern PetscErrorCode InitialConditions(Vec,AppCtx*);
extern PetscErrorCode RHSMatrixHeat(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode IFunctionHeat(TS,PetscReal,Vec,Vec,Vec,void*);
extern PetscErrorCode IJacobianHeat(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);

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
  }
  ierr = VecCopy(X,Xprev);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* NEW: activate new impl */
PetscErrorCode TSRollBackGenericActivate(TS ts)
{
  PetscFunctionBegin;
  if (!ts->setupcalled) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_WRONGSTATE,"Cannot activate generic rollback before calling TSSetUp()");
  if (ts->ops->rollback) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_WRONGSTATE,"Refusing to activate generic rollback for a TS implementation that already implements TSRollBack()");
  ts->ops->rollback = TSRollBack_Generic;
  PetscFunctionReturn(0);
}

/* NEW: destroy rollback state data */
PetscErrorCode TSRollBackGenericDestroy(TS ts)
{
  PetscErrorCode ierr;
  Vec            Xprev;

  ierr = PetscObjectQuery((PetscObject)ts,"RollBackGeneric_Xprev",(PetscObject*)(&Xprev));CHKERRQ(ierr);
  if (Xprev) {
    ierr = VecDestroy(&Xprev);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)ts,"RollBackGeneric_Xprev",NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* NEW: example of user post-step hook to roll back and stop */
PetscErrorCode PostStep_User(TS ts)
{
  PetscErrorCode ierr;
  PetscInt       stepNumber;

  PetscFunctionBegin;
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
  AppCtx         appctx;                 /* user-defined application context */
  TS             ts;                     /* timestepping context */
  Mat            A;                      /* matrix data structure */
  Vec            u;                      /* approximate solution vector */
  PetscReal      time_total_max = 100.0; /* default max total time */
  PetscInt       time_steps_max = 100;   /* default max timesteps */
  PetscErrorCode ierr;
  PetscInt       m;
  PetscMPIInt    size;
  PetscReal      dt;
  PetscBool      flg;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program and set problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_SELF,1,"This is a uniprocessor example only!");

  m    = 60;
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);

  appctx.m        = m;
  appctx.h        = 1.0/(m-1.0);

  ierr = VecCreateSeq(PETSC_COMM_SELF,m,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.solution);CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_SELF,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_LINEAR);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_SELF,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,m);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-use_ifunc",&flg,NULL);CHKERRQ(ierr);
  if (!flg) {
    appctx.A = NULL;
    ierr = PetscOptionsGetBool(NULL,NULL,"-time_dependent_rhs",&flg,NULL);CHKERRQ(ierr);
    if (flg) {
      /*
         For linear problems with a time-dependent f(u,t) in the equation
         u_t = f(u,t), the user provides the discretized right-hand-side
         as a time-dependent matrix.
      */
      ierr = TSSetRHSFunction(ts,NULL,TSComputeRHSFunctionLinear,&appctx);CHKERRQ(ierr);
      ierr = TSSetRHSJacobian(ts,A,A,RHSMatrixHeat,&appctx);CHKERRQ(ierr);
    } else {
      /*
         For linear problems with a time-independent f(u) in the equation
         u_t = f(u), the user provides the discretized right-hand-side
         as a matrix only once, and then sets the special Jacobian evaluation
         routine TSComputeRHSJacobianConstant() which will NOT recompute the Jacobian.
      */
      ierr = RHSMatrixHeat(ts,0.0,u,A,A,&appctx);CHKERRQ(ierr);
      ierr = TSSetRHSFunction(ts,NULL,TSComputeRHSFunctionLinear,&appctx);CHKERRQ(ierr);
      ierr = TSSetRHSJacobian(ts,A,A,TSComputeRHSJacobianConstant,&appctx);CHKERRQ(ierr);
    }
  } else {
    Mat J;

    ierr = RHSMatrixHeat(ts,0.0,u,A,A,&appctx);CHKERRQ(ierr);
    ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&J);CHKERRQ(ierr);
    ierr = TSSetIFunction(ts,NULL,IFunctionHeat,&appctx);CHKERRQ(ierr);
    ierr = TSSetIJacobian(ts,J,J,IJacobianHeat,&appctx);CHKERRQ(ierr);
    ierr = MatDestroy(&J);CHKERRQ(ierr);

    ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
    appctx.A = A;
    appctx.oshift = PETSC_MIN_REAL;
  }

  dt   = appctx.h*appctx.h/2.0;
  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);

  ierr = TSSetMaxSteps(ts,time_steps_max);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,time_total_max);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = InitialConditions(u,&appctx);CHKERRQ(ierr);

  /* NEW: add generic implementation for rollback (refuse to do it */
  ierr = TSSetSolution(ts,u);CHKERRQ(ierr); /* required for TSSetUp() */
  ierr = TSSetUp(ts);CHKERRQ(ierr);
  ierr = TSRollBackGenericActivate(ts);CHKERRQ(ierr);

  /* NEW: add prestep logic to keep previous state up-to-date */
  ierr = TSSetPreStep(ts,PreStep_RollBackGeneric);CHKERRQ(ierr);

  /* NEW: add poststep function to rollback and stop based on
    some criterion */
  ierr = TSSetPostStep(ts,PostStep_User);CHKERRQ(ierr);

  ierr = TSSolve(ts,NULL);CHKERRQ(ierr);

  /* NEW: free auxiliary data (could be integrated into TSDestroy with a flag check)*/
  ierr = TSRollBackGenericDestroy(ts);CHKERRQ(ierr);

  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.solution);CHKERRQ(ierr);
  ierr = MatDestroy(&appctx.A);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/* --------------------------------------------------------------------- */
/*
   InitialConditions - Computes the solution at the initial time.

   Input Parameter:
   u - uninitialized solution vector (global)
   appctx - user-defined application context

   Output Parameter:
   u - vector with solution at initial time (global)
*/
PetscErrorCode InitialConditions(Vec u,AppCtx *appctx)
{
  PetscScalar    *u_localptr,h = appctx->h;
  PetscErrorCode ierr;
  PetscInt       i;

  /*
    Get a pointer to vector data.
    - For default PETSc vectors, VecGetArray() returns a pointer to
      the data array.  Otherwise, the routine is implementation dependent.
    - You MUST call VecRestoreArray() when you no longer need access to
      the array.
    - Note that the Fortran interface to VecGetArray() differs from the
      C version.  See the users manual for details.
  */
  ierr = VecGetArray(u,&u_localptr);CHKERRQ(ierr);

  /*
     We initialize the solution array by simply writing the solution
     directly into the array locations.  Alternatively, we could use
     VecSetValues() or VecSetValuesLocal().
  */
  for (i=0; i<appctx->m; i++) u_localptr[i] = PetscSinScalar(PETSC_PI*i*6.*h) + 3.*PetscSinScalar(PETSC_PI*i*2.*h);

  /*
     Restore vector
  */
  ierr = VecRestoreArray(u,&u_localptr);CHKERRQ(ierr);

  return 0;
}

/* --------------------------------------------------------------------- */
/*
   RHSMatrixHeat - User-provided routine to compute the right-hand-side
   matrix for the heat equation.

   Input Parameters:
   ts - the TS context
   t - current time
   global_in - global input vector
   dummy - optional user-defined context, as set by TSetRHSJacobian()

   Output Parameters:
   AA - Jacobian matrix
   BB - optionally different preconditioning matrix
   str - flag indicating matrix structure

   Notes:
   Recall that MatSetValues() uses 0-based row and column numbers
   in Fortran as well as in C.
*/
PetscErrorCode RHSMatrixHeat(TS ts,PetscReal t,Vec X,Mat AA,Mat BB,void *ctx)
{
  Mat            A       = AA;                /* Jacobian matrix */
  AppCtx         *appctx = (AppCtx*)ctx;     /* user-defined application context */
  PetscInt       mstart  = 0;
  PetscInt       mend    = appctx->m;
  PetscErrorCode ierr;
  PetscInt       i,idx[3];
  PetscScalar    v[3],stwo = -2./(appctx->h*appctx->h),sone = -.5*stwo;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute entries for the locally owned part of the matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Set matrix rows corresponding to boundary data
  */

  mstart = 0;
  v[0]   = 1.0;
  ierr   = MatSetValues(A,1,&mstart,1,&mstart,v,INSERT_VALUES);CHKERRQ(ierr);
  mstart++;

  mend--;
  v[0] = 1.0;
  ierr = MatSetValues(A,1,&mend,1,&mend,v,INSERT_VALUES);CHKERRQ(ierr);

  /*
     Set matrix rows corresponding to interior data.  We construct the
     matrix one row at a time.
  */
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

PetscErrorCode IFunctionHeat(TS ts,PetscReal t,Vec X,Vec Xdot,Vec r,void *ctx)
{
  AppCtx         *appctx = (AppCtx*)ctx;     /* user-defined application context */
  PetscErrorCode ierr;

  ierr = MatMult(appctx->A,X,r);CHKERRQ(ierr);
  ierr = VecAYPX(r,-1.0,Xdot);CHKERRQ(ierr);
  return 0;
}

PetscErrorCode IJacobianHeat(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal s,Mat A,Mat B,void *ctx)
{
  AppCtx         *appctx = (AppCtx*)ctx;     /* user-defined application context */
  PetscErrorCode ierr;

  if (appctx->oshift == s) return 0;
  ierr = MatCopy(appctx->A,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatScale(A,-1);CHKERRQ(ierr);
  ierr = MatShift(A,s);CHKERRQ(ierr);
  ierr = MatCopy(A,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  appctx->oshift = s;
  return 0;
}
