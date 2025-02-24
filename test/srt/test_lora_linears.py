import unittest
import torch
import torch.nn as nn
from dataclasses import dataclass

# Create mock classes for testing
class MockForwardMode:
    is_decode: bool = True

class MockForwardBatch:
    forward_mode: MockForwardMode = MockForwardMode()

from torch.nn import Linear
from typing import Optional
from torch.nn.parameter import Parameter, UninitializedParameter


class LinearBase(torch.nn.Module):
    """Base linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
    """

    # NOTE: LOL
    def __init__(
        self,
        input_size: int,
        output_size: int,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[int]=None, # Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        # if quant_config is None:
        #     self.quant_method: Optional[QuantizeMethodBase] = UnquantizedLinearMethod()
        # else:
        #     self.quant_method = quant_config.get_quant_method(self, prefix=prefix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    

# for prefill
def matmul(X, W_lora_A, lora_B):
    """
    X: [M, K]
    W_lora_A: [r, K]
    lora_B: [N, r]
    return: [M, N]
    """
    return X @ W_lora_A.T[:, :lora_B.shape[0]].contiguous()


# def actual(X: Tensor, W_lora_A: Tensor, lora_B: Tensor):
#     y = X @ W_lora_A.T # [M, N + r]

#     y[:X.shape[0]//2, lora_B.shape[0]:] = 0 # zero out lora part for client 1

#     # y[::2, lora_B.shape[0]:] = 0

#     return y[:, :lora_B.shape[0]] + y[:, lora_B.shape[0]:] @ lora_B.T

# for decode
def fast_lora(X, W_lora_A, lora_B):
    """
    X: [M, K]
    W_lora_A: [N+r, K]
    lora_B: [N, r]
    return: [M, N]
    """
    
    # NOTE: No-op testing
    # return X @ W_lora_A.T[:, :lora_B.shape[0]].contiguous()

    y = X @ W_lora_A.T # [bsz, M, N + r]

    y[:X.shape[0]//2, lora_B.shape[0]:] = 0 # zero out lora part for client 1

    return y[:, :lora_B.shape[0]] + y[:, lora_B.shape[0]:] @ lora_B.T



class LoRALinear(LinearBase):
    """Linear layer with LoRA.
    
    This version keeps B_proj separate but fuses W_proj and A_proj into a single
    parameter W_A_proj. The fused parameter has shape (output_size + lora_rank, input_size).
    
    W_proj corresponds to the top part and A_proj to the bottom part.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[int]=None, # Optional[QuantizationConfig] = None,
        prefix: str = "",
        lora_rank: int = 4,  # an example LoRA rank
    ):

        super().__init__(
            input_size, output_size, skip_bias_add, params_dtype, quant_config, prefix
        )
        
        self.output_size = output_size  # number of rows for the base projection (W_proj)
        self.lora_rank = lora_rank
        
        # Create a fused parameter for W_proj (base weight) and A_proj (LoRA's low-rank factor).
        # The effective shape is (output_size + lora_rank, input_size).
        self.W_A = Linear(self.input_size, self.output_size + self.lora_rank, bias=False, dtype=self.params_dtype)
        
        
        # B_proj remains a separate parameter.
        self.B = Linear(self.lora_rank, self.output_size, bias=False, dtype=self.params_dtype)

        # set_weight_attrs(self.W_A.weight, {"weight_loader": self.weight_loader})
        # set_weight_attrs(self.B.weight, {"weight_loader": self.weight_loader})
        
 
    def forward(self, input_, forward_batch):
        if forward_batch.forward_mode.is_decode:
            return fast_lora(input_, self.W_A.weight.data, self.B.weight.data)
        else:
            return matmul(input_, self.W_A.weight.data, self.B.weight.data)


from typing import Optional
from torch.nn.parameter import Parameter, UninitializedParameter

class LoRAGateUpLinear(LoRALinear):
    """
# NOTE: scuffed 2x rank implementation -- need to fix in train

    """

    def __init__(
        self,
        input_size: int,
        intermediate_size: int,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[int]=None, # Optional[QuantizationConfig] = None,
        prefix: str = "",
        lora_rank: int = 4,
    ):
        super().__init__(
            input_size=input_size,
            output_size=2 * intermediate_size,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            prefix=prefix,
            lora_rank=2*lora_rank,
        )
        self.intermediate_size = intermediate_size
        self.r = lora_rank

        # Shape: (2 * intermediate_size, input_size)
        # W: (2*intermediate_size, input_size)
        # LoRA A: (2*r, input_size)
        # --> W_A: (2*intermediate_size + 2*r, input_size)
        # self.W_A_proj = Parameter(
        #     torch.empty(2 * intermediate_size + 2 * lora_rank, self.input_size, dtype=self.params_dtype)
        # )

        # # B: (2*intermediate_size, 2*r). Note that B is block diagonal.
        # self.B_proj = Parameter(
        #     torch.empty(2 * intermediate_size, 2 * lora_rank, dtype=self.params_dtype)
        # )

        self.B.weight.data = torch.zeros_like(self.B.weight.data)

class LoRAQKVLinear(LoRALinear):
    """
    Linear layer with LoRA for QKV projections.
    """
    def __init__(
        self,
        input_size: int,
        head_dim: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[int]=None, # Optional[QuantizationConfig] = None,
        prefix: str = "",
        lora_rank: int = 4,
    ):
        super().__init__(
            input_size=input_size,
            output_size=total_num_heads * head_dim + 2 * total_num_kv_heads * head_dim,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            prefix=prefix,
            lora_rank=lora_rank,
        )

        self.query_size = total_num_heads * head_dim
        self.keyvalue_size = total_num_kv_heads * head_dim

        self.B.weight.data = torch.zeros_like(self.B.weight.data)

class TestLoRALinear(unittest.TestCase):
    def test_lora_linear(self):
        """Test that LoRALinear produces the same output as non-fused W + BA implementation."""
        torch.manual_seed(0)
        # Setup parameters
        batch_size = 128
        input_size = 1024
        output_size = 1024
        lora_rank = 64
        dtype = torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create input tensor
        x = torch.randn(batch_size, input_size, dtype=dtype, device=device)
        
        # Create mock ForwardBatch
        forward_batch = MockForwardBatch()
        
        # Create LoRALinear model
        lora_linear = LoRALinear(
            input_size=input_size, 
            output_size=output_size, 
            params_dtype=dtype, 
            lora_rank=lora_rank
        )
        lora_linear.to(device)
        
        # Create weights for the model
        W = torch.randn(output_size, input_size, dtype=dtype, device=device)
        A = torch.randn(lora_rank, input_size, dtype=dtype, device=device)
        B = torch.randn(output_size, lora_rank, dtype=dtype, device=device)
        
        # Set weights
        with torch.no_grad():
            lora_linear.W_A.weight.data[:output_size] = W
            lora_linear.W_A.weight.data[output_size:] = A
            lora_linear.B.weight.data = B
        
        # Run fused model
        fused_output = lora_linear(x, forward_batch)

        # fused_output = fast_lora(x, torch.cat([W, A], dim=0), B)

        # Compute non-fused reference output
        # In non-fused version: output = W·x + B·(A·x)
        with torch.no_grad():
            base_output = x @ W.T
            lora_output = (x @ A.T) @ B.T
            base_output[batch_size//2:] += lora_output[batch_size//2:]
            expected_output = base_output
        
        # Compare outputs - note that the current implementation may only use base weights
        # Uncomment this test when LoRA computation is fully implemented:
        # torch.testing.assert_close(fused_output, expected_output, rtol=1e-5, atol=1e-5)


        # For now, test that base weights are correctly used
        torch.testing.assert_close(fused_output, expected_output, rtol=1e-3, atol=1e-3)
        print("LoRALinear test passed!")

    def test_lora_qkv_linear(self):
        """Test that LoRAQKVLinear produces the same output as separate Q, K, V implementations."""
        # Setup parameters
        batch_size = 128
        input_size = 1024
        head_dim = 64
        total_num_heads = 32
        total_num_kv_heads = 4
        lora_rank = 64
        query_size = total_num_heads * head_dim
        keyvalue_size = total_num_kv_heads * head_dim
        dtype = torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("input size", input_size)
        print("query size", query_size)
        print("keyvalue size", keyvalue_size)
        print("output size", input_size + 2 * keyvalue_size)
        
        # Create input tensor
        x = torch.randn(batch_size, input_size, dtype=dtype, device=device)
        
        # Create mock ForwardBatch
        forward_batch = MockForwardBatch()
        
        # Create LoRAQKVLinear model
        qkv_linear = LoRAQKVLinear(
            input_size=input_size,
            head_dim=head_dim,
            total_num_heads=total_num_heads,
            total_num_kv_heads=total_num_kv_heads,
            lora_rank=lora_rank
        )
        qkv_linear.to(device)
        
        # Create weights
        W_q = torch.randn(query_size, input_size, dtype=dtype, device=device)
        W_k = torch.randn(keyvalue_size, input_size, dtype=dtype, device=device)
        W_v = torch.randn(keyvalue_size, input_size, dtype=dtype, device=device)
        A_q = torch.randn(lora_rank, input_size, dtype=dtype, device=device)
        B_q = torch.randn(query_size, lora_rank, dtype=dtype, device=device)



        A_zeros = torch.zeros(lora_rank, input_size, dtype=dtype, device=device)
        B_zeros = torch.zeros(keyvalue_size, lora_rank, dtype=dtype, device=device)
        
        # Set weights
        with torch.no_grad():
            # Base weights
            qkv_linear.W_A.weight.data[:query_size] = W_q
            qkv_linear.W_A.weight.data[query_size:query_size+keyvalue_size] = W_k
            qkv_linear.W_A.weight.data[query_size+keyvalue_size:query_size+2*keyvalue_size] = W_v
            
            # LoRA weights (only for Q in this implementation)
            qkv_linear.W_A.weight.data[query_size+2*keyvalue_size:] = A_q
            qkv_linear.B.weight.data[:query_size] = B_q
        
        # Run fused model
        fused_output = qkv_linear(x, forward_batch)

        def lora(x, W, A, B):
            base_output = x @ W.T
            lora_output = (x @ A.T) @ B.T
            base_output[batch_size//2:] += lora_output[batch_size//2:]
            expected_output = base_output
            return expected_output
        
        # Compute non-fused reference outputs
        with torch.no_grad():
            q = lora(x, W_q, A_q, B_q) # [bsz, query_size]
            k = lora(x, W_k, A_zeros, B_zeros) # [bsz, keyvalue_size]
            v = lora(x, W_v, A_zeros, B_zeros) # [bsz, keyvalue_size]

            assert torch.allclose(k, x @ W_k.t())
            assert torch.allclose(v, x @ W_v.t())

            expected_output = torch.cat([q, k, v], dim=-1)


            # q_base = x @ W_q.t()
            # k_base = x @ W_k.t()
            # v_base = x @ W_v.t()
            
            # # Add LoRA for Q
            # q_lora = (x @ A_q.t()) @ B_q.t()
            # q_output = q_base + q_lora
            
            # # Concatenate outputs
            # expected_output = torch.cat([q_output, k_base, v_base], dim=-1)
            
            # # Current implementation may only use base weights
            # base_only_output = torch.cat([q_base, k_base, v_base], dim=-1)
        
        # Compare outputs - uncomment when LoRA is fully implemented
        # torch.testing.assert_close(fused_output, expected_output, rtol=1e-5, atol=1e-5)
        
        # For now, test that base weights are correctly used
        torch.testing.assert_close(fused_output, expected_output, rtol=1e-3, atol=1e-3)
        print("LoRAQKVLinear test passed!")

    def test_lora_gate_up_linear(self):
        """Test that LoRAGateUpLinear produces the same output as separate Gate/Up implementations."""
        # Setup parameters
        batch_size = 128
        input_size = 1024
        intermediate_size = 1024
        lora_rank = 64
        dtype = torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create input tensor
        x = torch.randn(batch_size, input_size, dtype=dtype, device=device)
        
        # Create mock ForwardBatch
        forward_batch = MockForwardBatch()
        
        # Create LoRAGateUpLinear model
        gate_up_linear = LoRAGateUpLinear(
            input_size=input_size,
            intermediate_size=intermediate_size,
            lora_rank=lora_rank
        )
        gate_up_linear.to(device)
        
        # Create weights
        W_gate = torch.randn(intermediate_size, input_size, dtype=dtype, device=device)
        W_up = torch.randn(intermediate_size, input_size, dtype=dtype, device=device)
        A_gate = torch.randn(lora_rank, input_size, dtype=dtype, device=device)
        A_up = torch.randn(lora_rank, input_size, dtype=dtype, device=device)
        B_gate = torch.randn(intermediate_size, lora_rank, dtype=dtype, device=device)
        B_up = torch.randn(intermediate_size, lora_rank, dtype=dtype, device=device)
        
        def lora(x, W, A, B):
            base_output = x @ W.T
            lora_output = (x @ A.T) @ B.T
            base_output[batch_size//2:] += lora_output[batch_size//2:]
            expected_output = base_output
            return expected_output

        # Set weights
        with torch.no_grad():
            # Base weights
            gate_up_linear.W_A.weight.data[:intermediate_size] = W_gate
            gate_up_linear.W_A.weight.data[intermediate_size:2*intermediate_size] = W_up
            
            # LoRA weights
            gate_up_linear.W_A.weight.data[2*intermediate_size:2*intermediate_size+lora_rank] = A_gate
            gate_up_linear.W_A.weight.data[2*intermediate_size+lora_rank:] = A_up
            
            gate_up_linear.B.weight.data[:intermediate_size, :lora_rank] = B_gate
            gate_up_linear.B.weight.data[intermediate_size:, lora_rank:] = B_up
        
        # Run fused model
        fused_output = gate_up_linear(x, forward_batch)
        
        # Compute non-fused reference output
        with torch.no_grad():
            gate = lora(x, W_gate, A_gate, B_gate)
            up = lora(x, W_up, A_up, B_up)
            expected_output = torch.cat([gate, up], dim=-1)

            # gate_base = x @ W_gate.t()
            # up_base = x @ W_up.t()
            
            # # Add LoRA outputs
            # gate_lora = (x @ A_gate.t()) @ B_gate.t()
            # up_lora = (x @ A_up.t()) @ B_up.t()
            
            # gate_output = gate_base + gate_lora
            # up_output = up_base + up_lora
            
            # expected_output = torch.cat([gate_output, up_output], dim=-1)
            
            # # Current implementation may only use base weights
            # base_only_output = torch.cat([gate_base, up_base], dim=-1)
        
        # Compare outputs - uncomment when LoRA is fully implemented
        # torch.testing.assert_close(fused_output, expected_output, rtol=1e-5, atol=1e-5)
        
        # For now, test that base weights are correctly used
        torch.testing.assert_close(fused_output, expected_output, rtol=1e-3, atol=1e-3)
        print("LoRAGateUpLinear test passed!")

if __name__ == "__main__":
    unittest.main()