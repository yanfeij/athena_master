//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
//
// This program is free software: you can redistribute and/or modify it under the terms
// of the GNU General Public License (GPL) as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
// PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
// You should have received a copy of GNU GPL in the file LICENSE included in the code
// distribution.  If not see <http://www.gnu.org/licenses/>.
//======================================================================================
//! \file tasklist.hpp
//  \brief task functions
//======================================================================================

// C/C++ headers
#include <iostream>   // cout, endl
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ classes headers
#include "athena.hpp"
#include "mesh.hpp"
#include "hydro/hydro.hpp"
#include "field/field.hpp"
#include "bvals/bvals.hpp"
#include "hydro/eos/eos.hpp"
#include "hydro/integrators/hydro_integrator.hpp"
#include "field/integrators/field_integrator.hpp"

// this class header
#include "tasklist.hpp"

//--------------------------------------------------------------------------------------
// TaskList constructor

TaskList::TaskList(Mesh *pm)
{
  pmy_mesh_ = pm;
  ntasks = 0;

  if (MAGNETIC_FIELDS_ENABLED) { // MHD

  // MHD predict
    AddTask(1,HYD_INT,1,NONE);
    if(pmy_mesh_->multilevel==true) { // SMR or AMR
      AddTask(1,FLX_SEND,1,HYD_INT);
      AddTask(1,FLX_RECV,1,HYD_INT);
      AddTask(1,HYD_SEND,1,FLX_RECV);
      AddTask(1,CALC_EMF,1,HYD_INT);
      AddTask(1,EMF_SEND,1,CALC_EMF);
      AddTask(1,EMF_RECV,1,EMF_SEND);
      AddTask(1,FLD_INT, 1,EMF_RECV);
    } else {
      AddTask(1,HYD_SEND,1,HYD_INT);
      AddTask(1,CALC_EMF,1,HYD_INT);
      AddTask(1,FLD_INT, 1,CALC_EMF);
    }
    AddTask(1,FLD_SEND,1,FLD_INT);
    AddTask(1,HYD_RECV,1,NONE);
    AddTask(1,HYD_BVAL,1,(HYD_RECV|HYD_INT));
    if(pmy_mesh_->multilevel==true) { // SMR or AMR
      AddTask(1,HYD_PROL,1,HYD_BVAL);
    }
    AddTask(1,FLD_RECV,1,NONE);
    AddTask(1,FLD_BVAL,1,(FLD_RECV|FLD_INT));
    if(pmy_mesh_->multilevel==true) {// SMR or AMR
      AddTask(1,FLD_PROL,1,FLD_BVAL);
      AddTask(1,CON2PRIM,1,(HYD_PROL|FLD_PROL));
    } else {
      AddTask(1,CON2PRIM,1,(HYD_BVAL|FLD_BVAL));
    }

  // MHD correct
    AddTask(2,HYD_INT,1,CON2PRIM);
    if(pmy_mesh_->multilevel==true) { // SMR or AMR
      AddTask(2,FLX_SEND,2,HYD_INT);
      AddTask(2,FLX_RECV,2,HYD_INT);
      AddTask(2,HYD_SEND,2,FLX_RECV);
      AddTask(2,CALC_EMF,2,HYD_INT);
      AddTask(2,EMF_SEND,2,CALC_EMF);
      AddTask(2,EMF_RECV,2,EMF_SEND);
      AddTask(2,FLD_INT, 2,EMF_RECV);
    } else {
      AddTask(2,HYD_SEND,2,HYD_INT);
      AddTask(2,CALC_EMF,2,HYD_INT);
      AddTask(2,FLD_INT, 2,CALC_EMF);
    }
    AddTask(2,FLD_SEND,2,FLD_INT);
    AddTask(2,HYD_RECV,1,CON2PRIM);
    AddTask(2,HYD_BVAL,2,(HYD_RECV|HYD_INT));
    if(pmy_mesh_->multilevel==true) { // SMR or AMR
      AddTask(2,HYD_PROL,2,HYD_BVAL);
    }
    AddTask(2,FLD_RECV,1,CON2PRIM);
    AddTask(2,FLD_BVAL,2,(FLD_RECV|FLD_INT));
    if(pmy_mesh_->multilevel==true) {// SMR or AMR
      AddTask(2,FLD_PROL,2,FLD_BVAL);
      AddTask(2,CON2PRIM,2,(HYD_PROL|FLD_PROL));
    } else { 
      AddTask(2,CON2PRIM,2,(HYD_BVAL|FLD_BVAL));
    }

  // Hydro predict
  } else {
    AddTask(1,HYD_INT,1,NONE);
    if(pmy_mesh_->multilevel==true) { // SMR or AMR
      AddTask(1,FLX_SEND,1,HYD_INT);
      AddTask(1,FLX_RECV,1,HYD_INT);
      AddTask(1,HYD_SEND,1,FLX_RECV);
    } else {
      AddTask(1,HYD_SEND,1,HYD_INT);
    }
    AddTask(1,HYD_RECV,1,NONE);  
    AddTask(1,HYD_BVAL,1,(HYD_RECV|HYD_INT));
    if(pmy_mesh_->multilevel==true) { // SMR or AMR
      AddTask(1,HYD_PROL,1,HYD_BVAL);
      AddTask(1,CON2PRIM,1,HYD_PROL);
    } else {
      AddTask(1,CON2PRIM,1,HYD_BVAL);
    }

  // Hydro correct
    AddTask(2,HYD_INT,1,CON2PRIM);
    if(pmy_mesh_->multilevel==true) { // SMR or AMR
      AddTask(2,FLX_SEND,2,HYD_INT);
      AddTask(2,FLX_RECV,2,HYD_INT);
      AddTask(2,HYD_SEND,2,FLX_RECV);
    } else {
      AddTask(2,HYD_SEND,2,HYD_INT);
    }
    AddTask(2,HYD_RECV,1,CON2PRIM);
    AddTask(2,HYD_BVAL,2,(HYD_RECV|HYD_INT));
    if(pmy_mesh_->multilevel==true) { // SMR or AMR
      AddTask(2,HYD_PROL,2,HYD_BVAL);
      AddTask(2,CON2PRIM,2,HYD_PROL);
    } else {
      AddTask(2,CON2PRIM,2,HYD_BVAL);
    }
  }

  // New timestep on mesh block
  AddTask(2,NEW_DT,2,CON2PRIM);

}

// destructor

TaskList::~TaskList()
{
}

namespace taskfunc {

enum task_status HydroIntegrate(MeshBlock *pmb, unsigned long int task_id, int step)
{
  Hydro *phydro=pmb->phydro;
  Field *pfield=pmb->pfield;

  if(step == 1) {
    phydro->u1 = phydro->u;
    phydro->pf_integrator->OneStep(pmb, phydro->u1, phydro->w, pfield->b,
                                   pfield->bcc, 1);
  } else if(step == 2) {
    phydro->pf_integrator->OneStep(pmb, phydro->u, phydro->w1, pfield->b1,
                                   pfield->bcc1, 2);
  } else {
    return task_failure;
  }

  return task_do_next;
}

enum task_status CalculateEMF(MeshBlock *pmb, unsigned long int task_id, int step)
{
  Hydro *phydro=pmb->phydro;
  Field *pfield=pmb->pfield;
  if(step == 1) {
    pfield->pint->ComputeCornerE(pmb, phydro->w, pfield->bcc);
  } else if(step == 2) {
    pfield->pint->ComputeCornerE(pmb, phydro->w1, pfield->bcc1);
  } else {
    return task_failure;
  }
  return task_do_next;
}

enum task_status FieldIntegrate(MeshBlock *pmb, unsigned long int task_id, int step)
{
  Hydro *phydro=pmb->phydro;
  Field *pfield=pmb->pfield;
  if(step == 1) {
    pfield->b1.x1f = pfield->b.x1f;
    pfield->b1.x2f = pfield->b.x2f;
    pfield->b1.x3f = pfield->b.x3f;
    pfield->pint->CT(pmb, pfield->b1, phydro->w, pfield->bcc, 1);
  } else if(step == 2) {
    pfield->pint->CT(pmb, pfield->b, phydro->w1, pfield->bcc1, 2);
  } else {
    return task_failure;
  }
  return task_do_next;
}

enum task_status HydroSend(MeshBlock *pmb, unsigned long int task_id, int step)
{
  Hydro *phydro=pmb->phydro;
  BoundaryValues *pbval=pmb->pbval;
  if(step == 1) {
    pbval->SendHydroBoundaryBuffers(phydro->u1,1);
  } else if(step == 2) {
    pbval->SendHydroBoundaryBuffers(phydro->u,0);
  } else {
    return task_failure;
  }
  return task_success;
}

enum task_status HydroReceive(MeshBlock *pmb, unsigned long int task_id, int step)
{
  Hydro *phydro=pmb->phydro;
  BoundaryValues *pbval=pmb->pbval;
  bool ret;
  if(step == 1) {
    ret=pbval->ReceiveHydroBoundaryBuffers(phydro->u1,1);
  } else if(step == 2) {
    ret=pbval->ReceiveHydroBoundaryBuffers(phydro->u,0);
  } else {
    return task_failure;
  }
  if(ret==true) {
    return task_success;
  } else {
    return task_failure;
  }
}

enum task_status FluxCorrectionSend(MeshBlock *pmb, unsigned long int task_id, int step)
{
  int flag;
  if(step == 1) {
    flag = 1;
  } else if(step == 2) {
    flag = 0;
  }
  pmb->pbval->SendFluxCorrection(flag);
  return task_success;
}

enum task_status FluxCorrectionReceive(MeshBlock *pmb, unsigned long int task_id, int step)
{
  Hydro *phydro=pmb->phydro;
  BoundaryValues *pbval=pmb->pbval;
  bool ret;
  if(step == 1) {
    ret=pbval->ReceiveFluxCorrection(phydro->u1,1);
  } else if(step == 2) {
    ret=pbval->ReceiveFluxCorrection(phydro->u,0);
  } else {
    return task_failure;
  }
  if(ret==true) {
    return task_do_next;
  } else {
    return task_failure;
  }
}

enum task_status HydroProlongation(MeshBlock *pmb, unsigned long int task_id, int step)
{
  Hydro *phydro=pmb->phydro;
  BoundaryValues *pbval=pmb->pbval;
  if(step == 1) {
    pbval->ProlongateHydroBoundaries(phydro->u1);
  } else if(step == 2) {
    pbval->ProlongateHydroBoundaries(phydro->u);
  } else {
    return task_failure;
  }
  return task_success;
}

enum task_status HydroPhysicalBoundary(MeshBlock *pmb, unsigned long int task_id, int step)
{
  Hydro *phydro=pmb->phydro;
  BoundaryValues *pbval=pmb->pbval;
  if(step == 1) {
    pbval->HydroPhysicalBoundaries(phydro->u1);
  } else if(step == 2) {
    pbval->HydroPhysicalBoundaries(phydro->u);
  } else {
    return task_failure;
  }
  return task_success;
}

enum task_status FieldSend(MeshBlock *pmb, unsigned long int task_id, int step)
{
  Field *pfield=pmb->pfield;
  BoundaryValues *pbval=pmb->pbval;
  if(step == 1) {
    pbval->SendFieldBoundaryBuffers(pfield->b1,1);
  } else if(step == 2) {
    pbval->SendFieldBoundaryBuffers(pfield->b,0);
  } else {
    return task_failure;
  }
  return task_success;
}

enum task_status FieldReceive(MeshBlock *pmb, unsigned long int task_id, int step)
{
  Field *pfield=pmb->pfield;
  BoundaryValues *pbval=pmb->pbval;
  bool ret;
  if(step == 1) {
    ret=pbval->ReceiveFieldBoundaryBuffers(pfield->b1,1);
  } else if(step == 2) {
    ret=pbval->ReceiveFieldBoundaryBuffers(pfield->b,0);
  } else {
    return task_failure;
  }
  if(ret==true) {
    return task_success;
  } else {
    return task_failure;
  }
}

enum task_status EMFCorrectionSend(MeshBlock *pmb, unsigned long int task_id, int step)
{
  int flag;
  if(step == 1) {
    flag = 1;
  } else if(step == 2) {
    flag = 0;
  }
  
  pmb->pbval->SendEMFCorrection(flag);
  return task_success;
}

enum task_status EMFCorrectionReceive(MeshBlock *pmb, unsigned long int task_id, int step)
{
  int flag;
  if(step == 1) {
    flag = 1;
  } else if(step == 2) {
    flag = 0;
  }
  BoundaryValues *pbval=pmb->pbval;
  if(pbval->ReceiveEMFCorrection(flag)==true) {
    return task_do_next;
  } else {
    return task_failure;
  }
}

enum task_status FieldProlongation(MeshBlock *pmb, unsigned long int task_id, int step)
{
  Field *pfield=pmb->pfield;
  BoundaryValues *pbval=pmb->pbval;
  if(step == 1) {
    pbval->ProlongateFieldBoundaries(pfield->b1);
  } else if(step == 2) {
    pbval->ProlongateFieldBoundaries(pfield->b);
  } else {
    return task_failure;
  }
  return task_success;
}

enum task_status FieldPhysicalBoundary(MeshBlock *pmb, unsigned long int task_id, int step)
{
  Field *pfield=pmb->pfield;
  BoundaryValues *pbval=pmb->pbval;
  if(step == 1) {
    pbval->FieldPhysicalBoundaries(pfield->b1);
  } else if(step == 2) {
    pbval->FieldPhysicalBoundaries(pfield->b);
  } else {
    return task_failure;
  }
  return task_success;
}

enum task_status Primitives(MeshBlock *pmb, unsigned long int task_id, int step)
{
  Hydro *phydro=pmb->phydro;
  Field *pfield=pmb->pfield;
  if(step == 1) {
    phydro->pf_eos->ConservedToPrimitive(phydro->u1, phydro->w, pfield->b1,
                                         phydro->w1, pfield->bcc1);
  } else if(step == 2) {
    phydro->pf_eos->ConservedToPrimitive(phydro->u, phydro->w1, pfield->b,
                                         phydro->w, pfield->bcc);
  } else {
    return task_failure;
  }
  return task_success;
}

enum task_status NewBlockTimeStep(MeshBlock *pmb, unsigned long int task_id, int step)
{
  pmb->phydro->NewBlockTimeStep(pmb);
  return task_success;
}

} // namespace task


void TaskList::AddTask(int stp_t,unsigned long int id,int stp_d,unsigned long int dep)
{
  task_list_[ntasks].step_of_task=stp_t;
  task_list_[ntasks].step_of_depend=stp_d;
  task_list_[ntasks].task_id=id;
  task_list_[ntasks].dependency=dep;

  switch((id))
  {
  case (HYD_INT):
    task_list_[ntasks].TaskFunc=taskfunc::HydroIntegrate;
    break;

  case (CALC_EMF):
    task_list_[ntasks].TaskFunc=taskfunc::CalculateEMF;
    break;

  case (FLD_INT):
    task_list_[ntasks].TaskFunc=taskfunc::FieldIntegrate;
    break;

  case (HYD_SEND):
    task_list_[ntasks].TaskFunc=taskfunc::HydroSend;
    break;

  case (HYD_RECV):
    task_list_[ntasks].TaskFunc=taskfunc::HydroReceive;
    break;

  case (FLX_SEND):
    task_list_[ntasks].TaskFunc=taskfunc::FluxCorrectionSend;
    break;

  case (FLX_RECV):
    task_list_[ntasks].TaskFunc=taskfunc::FluxCorrectionReceive;
    break;

  case (HYD_PROL):
    task_list_[ntasks].TaskFunc=taskfunc::HydroProlongation;
    break;

  case (HYD_BVAL):
    task_list_[ntasks].TaskFunc=taskfunc::HydroPhysicalBoundary;
    break;

  case (FLD_SEND):
    task_list_[ntasks].TaskFunc=taskfunc::FieldSend;
    break;

  case (FLD_RECV):
    task_list_[ntasks].TaskFunc=taskfunc::FieldReceive;
    break;

  case (EMF_SEND):
    task_list_[ntasks].TaskFunc=taskfunc::EMFCorrectionSend;
    break;

  case (EMF_RECV):
    task_list_[ntasks].TaskFunc=taskfunc::EMFCorrectionReceive;
    break;

  case (FLD_PROL):
    task_list_[ntasks].TaskFunc=taskfunc::FieldProlongation;
    break;

  case (FLD_BVAL):
    task_list_[ntasks].TaskFunc=taskfunc::FieldPhysicalBoundary;
    break;

  case (CON2PRIM):
    task_list_[ntasks].TaskFunc=taskfunc::Primitives;
    break;

  case (NEW_DT):
    task_list_[ntasks].TaskFunc=taskfunc::NewBlockTimeStep;
    break;

  default:
    std::stringstream msg;
    msg << "### FATAL ERROR in AddTask" << std::endl
        << "Invalid Task "<< id << " is specified" << std::endl;
    throw std::runtime_error(msg.str().c_str());
  
  }
  ntasks++;
  return;
}

//--------------------------------------------------------------------------------------
//! \fn
//  \brief process one task (if possible), return tasklist_status

enum tasklist_status TaskList::DoOneTask(MeshBlock *pmb) {
  int skip=0;
  enum task_status ret;
  std::stringstream msg;

  if(pmb->ntodo==0) return tl_nothing;

  for(int i=pmb->firsttask; i<ntasks; i++) {
    Task &ti=task_list_[i];
    if((ti.task_id & pmb->finished_tasks[ti.step_of_task])==0L) { // task not done
      // check if dependency clear
      if (((ti.dependency & pmb->finished_tasks[ti.step_of_depend]) == ti.dependency)) {
        ret=ti.TaskFunc(pmb,ti.task_id,ti.step_of_task);
        if(ret!=task_failure) { // success
          pmb->ntodo--;
          pmb->finished_tasks[ti.step_of_task] |= ti.task_id;
          if(skip==0)
            pmb->firsttask++;
          if(pmb->ntodo==0)
            return tl_complete;
          if(ret==task_do_next) continue;
          return tl_running;
        }
      }
      skip++; // increment number of tasks processed
    } else if(skip==0) // task is done and at the top of the list
      pmb->firsttask++;
  }
  return tl_stuck; // there are still tasks to do but nothing can be done now
}
