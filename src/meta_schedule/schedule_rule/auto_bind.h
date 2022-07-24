/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#ifndef TVM_META_SCHEDULE_SCHEDULE_RULE_AUTO_BIND_H_
#define TVM_META_SCHEDULE_SCHEDULE_RULE_AUTO_BIND_H_

#include "../utils.h"

namespace tvm {
namespace meta_schedule {

/*!
 * \brief Bind the given block if it is not bound to blockIdx or threadIdx.
 * \param sch The schedule.
 * \param block The block to be bound.
 * \param max_threadblocks The maximum number of threadblocks allowed.
 * \param max_threads The maximum number of threads allowed.
 * \param get_factor A function that returns the tiling factor.
 */
void BindBlockThreadIdx(const tir::Schedule& sch, const tir::BlockRV& block,
                        int64_t max_threadblocks, int64_t max_threads_per_block,
                        std::function<tir::ExprRV(int64_t max_extent)> get_factor);

/*!
 * \brief Given candidates of thread_extents, make a sampler that use `sch->SampleCategorical`
 * to return a random thread extent.
 * \param sch The schedule
 * \param thread_extents The candidate thread extents.
 * \return A sampler that returns a random thread extent.
 */
std::function<tir::ExprRV(int64_t max_extent)> MakeFactorSampler(tir::Schedule sch,
                                                                 Array<Integer> thread_extents);

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_SCHEDULE_RULE_AUTO_BIND_H_
