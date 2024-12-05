#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 1, 1] subgroup_size = 64>
module attributes {transform.with_named_sequence} {
  stream.executable private @base_attention {
    stream.executable.export public @base_attention workgroups() -> (index, index, index) {
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c40 = arith.constant 40 : index
      stream.return %c8, %c1, %c40 : index, index, index
    }
    builtin.module {
      func.func @base_attention(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding) attributes {translation_info = #translation} {
        %cst = arith.constant dense<1.000000e+00> : vector<1xf32>
        %c64_i32 = arith.constant 64 : i32
        %c32_i32 = arith.constant 32 : i32
        %c56 = arith.constant 56 : index
        %c48 = arith.constant 48 : index
        %c40 = arith.constant 40 : index
        %c24 = arith.constant 24 : index
        %c16 = arith.constant 16 : index
        %c8 = arith.constant 8 : index
        %c4 = arith.constant 4 : index
        %c128 = arith.constant 128 : index
        %c64 = arith.constant 64 : index
        %c1 = arith.constant 1 : index
        %c32 = arith.constant 32 : index
        %c0 = arith.constant 0 : index
        %cst_0 = arith.constant dense<-1.000000e+06> : vector<1xf32>
        %cst_1 = arith.constant dense<0.000000e+00> : vector<1xf32>
        %cst_2 = arith.constant dense<0.000000e+00> : vector<16xf32>
        %workgroup_id_0 = stream.dispatch.workgroup.id[0] : index
        %workgroup_id_1 = stream.dispatch.workgroup.id[1] : index
        %workgroup_id_2 = stream.dispatch.workgroup.id[2] : index
        %thread_id_x = gpu.thread_id  x
        %thread_id_y = gpu.thread_id  y
        %thread_id_z = gpu.thread_id  z
        %alloc = memref.alloc() : memref<1x64x68xf16, #gpu.address_space<workgroup>>
        %alloc_3 = memref.alloc() : memref<1x64x68xf16, #gpu.address_space<workgroup>>
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<40x1024x64xf16, strided<[65536, 64, 1], offset: ?>>
        %1 = arith.divsi %thread_id_x, %c64 : index
        %2 = arith.muli %1, %c32 overflow<nsw, nuw> : index
        %3 = arith.muli %workgroup_id_0, %c128 overflow<nsw, nuw> : index
        %4 = arith.remsi %thread_id_x, %c32 : index
        %5 = arith.addi %4, %3 overflow<nsw, nuw> : index
        %6 = arith.addi %5, %2 overflow<nsw, nuw> : index
        %7 = arith.remsi %thread_id_x, %c64 : index
        %8 = arith.divsi %7, %c32 : index
        %9 = arith.muli %8, %c4 overflow<nsw, nuw> : index
        %10 = vector.load %0[%workgroup_id_2, %6, %9] : memref<40x1024x64xf16, strided<[65536, 64, 1], offset: ?>>, vector<4xf16>
        %11 = arith.addi %9, %c8 overflow<nsw, nuw> : index
        %12 = vector.load %0[%workgroup_id_2, %6, %11] : memref<40x1024x64xf16, strided<[65536, 64, 1], offset: ?>>, vector<4xf16>
        %13 = arith.addi %9, %c16 overflow<nsw, nuw> : index
        %14 = vector.load %0[%workgroup_id_2, %6, %13] : memref<40x1024x64xf16, strided<[65536, 64, 1], offset: ?>>, vector<4xf16>
        %15 = arith.addi %9, %c24 overflow<nsw, nuw> : index
        %16 = vector.load %0[%workgroup_id_2, %6, %15] : memref<40x1024x64xf16, strided<[65536, 64, 1], offset: ?>>, vector<4xf16>
        %17 = arith.addi %9, %c32 overflow<nsw, nuw> : index
        %18 = vector.load %0[%workgroup_id_2, %6, %17] : memref<40x1024x64xf16, strided<[65536, 64, 1], offset: ?>>, vector<4xf16>
        %19 = arith.addi %9, %c40 overflow<nsw, nuw> : index
        %20 = vector.load %0[%workgroup_id_2, %6, %19] : memref<40x1024x64xf16, strided<[65536, 64, 1], offset: ?>>, vector<4xf16>
        %21 = arith.addi %9, %c48 overflow<nsw, nuw> : index
        %22 = vector.load %0[%workgroup_id_2, %6, %21] : memref<40x1024x64xf16, strided<[65536, 64, 1], offset: ?>>, vector<4xf16>
        %23 = arith.addi %9, %c56 overflow<nsw, nuw> : index
        %24 = vector.load %0[%workgroup_id_2, %6, %23] : memref<40x1024x64xf16, strided<[65536, 64, 1], offset: ?>>, vector<4xf16>
        %25 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<40x1024x64xf16, strided<[65536, 64, 1], offset: ?>>
        %26 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<40x64x1024xf16, strided<[65536, 1024, 1], offset: ?>>
        %27 = arith.muli %thread_id_y, %c32 overflow<nsw, nuw> : index
        %28 = arith.muli %thread_id_z, %c32 overflow<nsw, nuw> : index
        %29 = arith.divsi %thread_id_x, %c8 : index
        %30 = arith.addi %29, %28 overflow<nsw, nuw> : index
        %31 = arith.addi %30, %27 overflow<nsw, nuw> : index
        %32 = arith.remsi %31, %c64 : index
        %33 = arith.remsi %thread_id_x, %c8 : index
        %34 = arith.muli %33, %c8 overflow<nsw, nuw> : index
        %35 = arith.addi %31, %c32 overflow<nsw, nuw> : index
        %36 = arith.remsi %35, %c64 : index
        %37 = arith.addi %4, %c32 overflow<nsw, nuw> : index
        %38 = arith.muli %workgroup_id_1, %c64 overflow<nsw, nuw> : index
        %39 = arith.addi %32, %38 overflow<nsw, nuw> : index
        %40 = arith.addi %36, %38 overflow<nsw, nuw> : index
        %41 = arith.muli %thread_id_y, %c64 overflow<nsw, nuw> : index
        %42 = arith.addi %4, %41 overflow<nsw, nuw> : index
        %43 = arith.addi %42, %c32 overflow<nsw, nuw> : index
        %44:4 = scf.for %arg4 = %c0 to %c16 step %c1 iter_args(%arg5 = %cst_0, %arg6 = %cst_1, %arg7 = %cst_2, %arg8 = %cst_2) -> (vector<1xf32>, vector<1xf32>, vector<16xf32>, vector<16xf32>) {
          %70 = arith.muli %arg4, %c64 overflow<nsw, nuw> : index
          %71 = arith.addi %32, %70 overflow<nsw, nuw> : index
          %72 = vector.load %25[%workgroup_id_2, %71, %34] : memref<40x1024x64xf16, strided<[65536, 64, 1], offset: ?>>, vector<8xf16>
          vector.store %72, %alloc[%c0, %32, %34] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %73 = arith.addi %36, %70 overflow<nsw, nuw> : index
          %74 = vector.load %25[%workgroup_id_2, %73, %34] : memref<40x1024x64xf16, strided<[65536, 64, 1], offset: ?>>, vector<8xf16>
          vector.store %74, %alloc[%c0, %36, %34] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          amdgpu.lds_barrier
          %75 = vector.load %alloc[%c0, %4, %9] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %76 = vector.load %alloc[%c0, %37, %9] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %77 = vector.load %alloc[%c0, %37, %11] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %78 = vector.load %alloc[%c0, %37, %13] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %79 = vector.load %alloc[%c0, %37, %15] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %80 = vector.load %alloc[%c0, %37, %17] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %81 = vector.load %alloc[%c0, %37, %19] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %82 = vector.load %alloc[%c0, %37, %21] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %83 = vector.load %alloc[%c0, %37, %23] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %84 = vector.load %alloc[%c0, %4, %11] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %85 = vector.load %alloc[%c0, %4, %13] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %86 = vector.load %alloc[%c0, %4, %15] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %87 = vector.load %alloc[%c0, %4, %17] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %88 = vector.load %alloc[%c0, %4, %19] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %89 = vector.load %alloc[%c0, %4, %21] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %90 = vector.load %alloc[%c0, %4, %23] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %91 = amdgpu.mfma %75 * %10 + %cst_2 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %92 = amdgpu.mfma %76 * %10 + %cst_2 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %93 = amdgpu.mfma %77 * %12 + %92 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %94 = amdgpu.mfma %78 * %14 + %93 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %95 = amdgpu.mfma %79 * %16 + %94 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %96 = amdgpu.mfma %80 * %18 + %95 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %97 = amdgpu.mfma %81 * %20 + %96 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %98 = amdgpu.mfma %82 * %22 + %97 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %99 = amdgpu.mfma %83 * %24 + %98 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %100 = amdgpu.mfma %84 * %12 + %91 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %101 = amdgpu.mfma %85 * %14 + %100 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %102 = amdgpu.mfma %86 * %16 + %101 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %103 = amdgpu.mfma %87 * %18 + %102 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %104 = amdgpu.mfma %88 * %20 + %103 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %105 = amdgpu.mfma %89 * %22 + %104 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %106 = amdgpu.mfma %90 * %24 + %105 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %107 = arith.maximumf %106, %99 : vector<16xf32>
          %108 = vector.extract_strided_slice %107 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %109 = vector.extract_strided_slice %107 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %110 = arith.maximumf %108, %109 : vector<1xf32>
          %111 = vector.extract_strided_slice %107 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %112 = arith.maximumf %110, %111 : vector<1xf32>
          %113 = vector.extract_strided_slice %107 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %114 = arith.maximumf %112, %113 : vector<1xf32>
          %115 = vector.extract_strided_slice %107 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %116 = arith.maximumf %114, %115 : vector<1xf32>
          %117 = vector.extract_strided_slice %107 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %118 = arith.maximumf %116, %117 : vector<1xf32>
          %119 = vector.extract_strided_slice %107 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %120 = arith.maximumf %118, %119 : vector<1xf32>
          %121 = vector.extract_strided_slice %107 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %122 = arith.maximumf %120, %121 : vector<1xf32>
          %123 = vector.extract_strided_slice %107 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %124 = arith.maximumf %122, %123 : vector<1xf32>
          %125 = vector.extract_strided_slice %107 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %126 = arith.maximumf %124, %125 : vector<1xf32>
          %127 = vector.extract_strided_slice %107 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %128 = arith.maximumf %126, %127 : vector<1xf32>
          %129 = vector.extract_strided_slice %107 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %130 = arith.maximumf %128, %129 : vector<1xf32>
          %131 = vector.extract_strided_slice %107 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %132 = arith.maximumf %130, %131 : vector<1xf32>
          %133 = vector.extract_strided_slice %107 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %134 = arith.maximumf %132, %133 : vector<1xf32>
          %135 = vector.extract_strided_slice %107 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %136 = arith.maximumf %134, %135 : vector<1xf32>
          %137 = vector.extract_strided_slice %107 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %138 = arith.maximumf %136, %137 : vector<1xf32>
          %139 = vector.extract %138[0] : f32 from vector<1xf32>
          %shuffleResult, %valid = gpu.shuffle  xor %139, %c32_i32, %c64_i32 : f32
          %140 = vector.broadcast %shuffleResult : f32 to vector<1xf32>
          %141 = arith.maximumf %138, %140 : vector<1xf32>
          %142 = arith.maximumf %arg5, %141 : vector<1xf32>
          %143 = arith.subf %arg5, %142 : vector<1xf32>
          %144 = math.exp2 %143 : vector<1xf32>
          %145 = vector.extract %142[0] : f32 from vector<1xf32>
          %146 = vector.splat %145 : vector<16xf32>
          %147 = arith.subf %106, %146 : vector<16xf32>
          %148 = arith.subf %99, %146 : vector<16xf32>
          %149 = math.exp2 %147 : vector<16xf32>
          %150 = math.exp2 %148 : vector<16xf32>
          %151 = arith.mulf %arg6, %144 : vector<1xf32>
          %152 = arith.addf %149, %150 : vector<16xf32>
          %153 = vector.extract_strided_slice %152 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %154 = vector.extract_strided_slice %152 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %155 = arith.addf %153, %154 : vector<1xf32>
          %156 = vector.extract_strided_slice %152 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %157 = arith.addf %155, %156 : vector<1xf32>
          %158 = vector.extract_strided_slice %152 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %159 = arith.addf %157, %158 : vector<1xf32>
          %160 = vector.extract_strided_slice %152 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %161 = arith.addf %159, %160 : vector<1xf32>
          %162 = vector.extract_strided_slice %152 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %163 = arith.addf %161, %162 : vector<1xf32>
          %164 = vector.extract_strided_slice %152 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %165 = arith.addf %163, %164 : vector<1xf32>
          %166 = vector.extract_strided_slice %152 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %167 = arith.addf %165, %166 : vector<1xf32>
          %168 = vector.extract_strided_slice %152 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %169 = arith.addf %167, %168 : vector<1xf32>
          %170 = vector.extract_strided_slice %152 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %171 = arith.addf %169, %170 : vector<1xf32>
          %172 = vector.extract_strided_slice %152 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %173 = arith.addf %171, %172 : vector<1xf32>
          %174 = vector.extract_strided_slice %152 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %175 = arith.addf %173, %174 : vector<1xf32>
          %176 = vector.extract_strided_slice %152 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %177 = arith.addf %175, %176 : vector<1xf32>
          %178 = vector.extract_strided_slice %152 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %179 = arith.addf %177, %178 : vector<1xf32>
          %180 = vector.extract_strided_slice %152 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %181 = arith.addf %179, %180 : vector<1xf32>
          %182 = vector.extract_strided_slice %152 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %183 = arith.addf %181, %182 : vector<1xf32>
          %184 = vector.extract %183[0] : f32 from vector<1xf32>
          %shuffleResult_4, %valid_5 = gpu.shuffle  xor %184, %c32_i32, %c64_i32 : f32
          %185 = vector.broadcast %shuffleResult_4 : f32 to vector<1xf32>
          %186 = arith.addf %183, %185 : vector<1xf32>
          %187 = arith.addf %151, %186 : vector<1xf32>
          %188 = arith.truncf %149 : vector<16xf32> to vector<16xf16>
          %189 = arith.truncf %150 : vector<16xf32> to vector<16xf16>
          %190 = arith.addi %70, %34 overflow<nsw, nuw> : index
          %191 = vector.load %26[%workgroup_id_2, %39, %190] : memref<40x64x1024xf16, strided<[65536, 1024, 1], offset: ?>>, vector<8xf16>
          amdgpu.lds_barrier
          vector.store %191, %alloc_3[%c0, %32, %34] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %192 = vector.load %26[%workgroup_id_2, %40, %190] : memref<40x64x1024xf16, strided<[65536, 1024, 1], offset: ?>>, vector<8xf16>
          vector.store %192, %alloc_3[%c0, %36, %34] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          amdgpu.lds_barrier
          %193 = vector.load %alloc_3[%c0, %42, %9] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %194 = vector.load %alloc_3[%c0, %42, %11] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %195 = vector.load %alloc_3[%c0, %42, %13] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %196 = vector.load %alloc_3[%c0, %42, %15] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %197 = vector.load %alloc_3[%c0, %42, %17] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %198 = vector.load %alloc_3[%c0, %42, %19] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %199 = vector.load %alloc_3[%c0, %42, %21] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %200 = vector.load %alloc_3[%c0, %42, %23] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %201 = vector.load %alloc_3[%c0, %43, %9] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %202 = vector.load %alloc_3[%c0, %43, %11] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %203 = vector.load %alloc_3[%c0, %43, %13] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %204 = vector.load %alloc_3[%c0, %43, %15] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %205 = vector.load %alloc_3[%c0, %43, %17] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %206 = vector.load %alloc_3[%c0, %43, %19] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %207 = vector.load %alloc_3[%c0, %43, %21] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %208 = vector.load %alloc_3[%c0, %43, %23] : memref<1x64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %209 = vector.extract %144[0] : f32 from vector<1xf32>
          %210 = vector.splat %209 : vector<16xf32>
          %211 = arith.mulf %arg7, %210 : vector<16xf32>
          %212 = arith.mulf %arg8, %210 : vector<16xf32>
          %213 = vector.extract_strided_slice %188 {offsets = [0], sizes = [4], strides = [1]} : vector<16xf16> to vector<4xf16>
          %214 = vector.extract_strided_slice %188 {offsets = [4], sizes = [4], strides = [1]} : vector<16xf16> to vector<4xf16>
          %215 = vector.extract_strided_slice %188 {offsets = [8], sizes = [4], strides = [1]} : vector<16xf16> to vector<4xf16>
          %216 = vector.extract_strided_slice %188 {offsets = [12], sizes = [4], strides = [1]} : vector<16xf16> to vector<4xf16>
          %217 = vector.extract_strided_slice %189 {offsets = [0], sizes = [4], strides = [1]} : vector<16xf16> to vector<4xf16>
          %218 = vector.extract_strided_slice %189 {offsets = [4], sizes = [4], strides = [1]} : vector<16xf16> to vector<4xf16>
          %219 = vector.extract_strided_slice %189 {offsets = [8], sizes = [4], strides = [1]} : vector<16xf16> to vector<4xf16>
          %220 = vector.extract_strided_slice %189 {offsets = [12], sizes = [4], strides = [1]} : vector<16xf16> to vector<4xf16>
          %221 = amdgpu.mfma %193 * %213 + %211 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %222 = amdgpu.mfma %194 * %214 + %221 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %223 = amdgpu.mfma %195 * %215 + %222 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %224 = amdgpu.mfma %196 * %216 + %223 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %225 = amdgpu.mfma %197 * %217 + %224 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %226 = amdgpu.mfma %198 * %218 + %225 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %227 = amdgpu.mfma %199 * %219 + %226 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %228 = amdgpu.mfma %200 * %220 + %227 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %229 = amdgpu.mfma %201 * %213 + %212 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %230 = amdgpu.mfma %202 * %214 + %229 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %231 = amdgpu.mfma %203 * %215 + %230 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %232 = amdgpu.mfma %204 * %216 + %231 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %233 = amdgpu.mfma %205 * %217 + %232 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %234 = amdgpu.mfma %206 * %218 + %233 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %235 = amdgpu.mfma %207 * %219 + %234 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %236 = amdgpu.mfma %208 * %220 + %235 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          scf.yield %142, %187, %228, %236 : vector<1xf32>, vector<1xf32>, vector<16xf32>, vector<16xf32>
        }
        %45 = arith.divf %cst, %44#1 : vector<1xf32>
        %46 = vector.extract %45[0] : f32 from vector<1xf32>
        %47 = vector.splat %46 : vector<16xf32>
        %48 = arith.mulf %44#2, %47 : vector<16xf32>
        %49 = arith.mulf %44#3, %47 : vector<16xf32>
        %50 = vector.extract_strided_slice %48 {offsets = [0], sizes = [4], strides = [1]} : vector<16xf32> to vector<4xf32>
        %51 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<40x1024x64xf32, strided<[65536, 64, 1], offset: ?>>
        %52 = arith.muli %thread_id_y, %c64 overflow<nsw, nuw> : index
        %53 = arith.muli %workgroup_id_1, %c64 overflow<nsw, nuw> : index
        %54 = arith.addi %53, %52 overflow<nsw, nuw> : index
        %55 = arith.addi %54, %9 overflow<nsw, nuw> : index
        vector.store %50, %51[%workgroup_id_2, %6, %55] : memref<40x1024x64xf32, strided<[65536, 64, 1], offset: ?>>, vector<4xf32>
        %56 = vector.extract_strided_slice %48 {offsets = [4], sizes = [4], strides = [1]} : vector<16xf32> to vector<4xf32>
        %57 = arith.addi %55, %c8 overflow<nsw, nuw> : index
        vector.store %56, %51[%workgroup_id_2, %6, %57] : memref<40x1024x64xf32, strided<[65536, 64, 1], offset: ?>>, vector<4xf32>
        %58 = vector.extract_strided_slice %48 {offsets = [8], sizes = [4], strides = [1]} : vector<16xf32> to vector<4xf32>
        %59 = arith.addi %55, %c16 overflow<nsw, nuw> : index
        vector.store %58, %51[%workgroup_id_2, %6, %59] : memref<40x1024x64xf32, strided<[65536, 64, 1], offset: ?>>, vector<4xf32>
        %60 = vector.extract_strided_slice %48 {offsets = [12], sizes = [4], strides = [1]} : vector<16xf32> to vector<4xf32>
        %61 = arith.addi %55, %c24 overflow<nsw, nuw> : index
        vector.store %60, %51[%workgroup_id_2, %6, %61] : memref<40x1024x64xf32, strided<[65536, 64, 1], offset: ?>>, vector<4xf32>
        %62 = vector.extract_strided_slice %49 {offsets = [0], sizes = [4], strides = [1]} : vector<16xf32> to vector<4xf32>
        %63 = arith.addi %55, %c32 overflow<nsw, nuw> : index
        vector.store %62, %51[%workgroup_id_2, %6, %63] : memref<40x1024x64xf32, strided<[65536, 64, 1], offset: ?>>, vector<4xf32>
        %64 = vector.extract_strided_slice %49 {offsets = [4], sizes = [4], strides = [1]} : vector<16xf32> to vector<4xf32>
        %65 = arith.addi %55, %c40 overflow<nsw, nuw> : index
        vector.store %64, %51[%workgroup_id_2, %6, %65] : memref<40x1024x64xf32, strided<[65536, 64, 1], offset: ?>>, vector<4xf32>
        %66 = vector.extract_strided_slice %49 {offsets = [8], sizes = [4], strides = [1]} : vector<16xf32> to vector<4xf32>
        %67 = arith.addi %55, %c48 overflow<nsw, nuw> : index
        vector.store %66, %51[%workgroup_id_2, %6, %67] : memref<40x1024x64xf32, strided<[65536, 64, 1], offset: ?>>, vector<4xf32>
        %68 = vector.extract_strided_slice %49 {offsets = [12], sizes = [4], strides = [1]} : vector<16xf32> to vector<4xf32>
        %69 = arith.addi %55, %c56 overflow<nsw, nuw> : index
        vector.store %68, %51[%workgroup_id_2, %6, %69] : memref<40x1024x64xf32, strided<[65536, 64, 1], offset: ?>>, vector<4xf32>
        return
      }
    }
  }
  func.func @isolated_benchmark(%arg0: tensor<40x1024x64xf16>, %arg1: tensor<40x1024x64xf16>, %arg2: tensor<40x64x1024xf16>, %arg3: tensor<40x1024x64xf32>) -> tensor<40x1024x64xf32> {
    %0 = flow.dispatch @base_attention::@base_attention(%arg0, %arg1, %arg2, %arg3) : (tensor<40x1024x64xf16>, tensor<40x1024x64xf16>, tensor<40x64x1024xf16>, tensor<40x1024x64xf32>) -> %arg3
    return %0 : tensor<40x1024x64xf32>
  }
}
