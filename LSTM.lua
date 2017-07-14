local THNN = require 'nn.THNN'
local LSTM, parent = torch.class('nn.LSTM', 'nn.Module')

--[[
input:  X(t), H(t-1), C(t-1)
output: H(t), C(t)
F(t) = sig{Whf*H(t-1) + Wxf*X(t)}
I(t) = sig{Whi*H(t-1) + Wxi*X(t)}
O(t) = sig{Who*H(t-1) + Wxo*X(t)}
C_(t) = sig{Whc*H(t-1) + Wxc*X(t)}
C(t) = F(t) .* C(t-1) + I(t) .* C_(t)
H(t) = O(t) .* tan{C(t)}
--]]

function LSTM:__init(inputSize, hiddenSize, bias)
   parent.__init(self)
   outputSize = 4 * hiddenSize
   local bias = ((bias == nil) and true) or bias
   self.output_c = torch.Tensor()
   self.output_h = torch.Tensor()
   self.output = {self.output_c, self.output_h}
   self.grad_in_c = torch.Tensor()
   self.grad_in_h = torch.Tensor()
   self.grad_in_x = torch.Tensor()
   self.gradInput = {self.grad_in_c, self.grad_in_h, self.grad_in_x}
   self.weight_h = torch.Tensor(hiddenSize, outputSize)
   self.weight_x = torch.Tensor(inputSize,  outputSize)
   self.grad_weight_h = torch.Tensor(hiddenSize, outputSize)
   self.grad_weight_x = torch.Tensor(inputSize,  outputSize)
   if bias then
      self.bias_h      = torch.Tensor(outputSize)
      self.grad_bias_h = torch.Tensor(outputSize)
      self.bias_x      = torch.Tensor(outputSize)
      self.grad_bias_x = torch.Tensor(outputSize)
   end
   self:reset()
end

--to compare, initalize weight for h and x
--TODO, combineing codes
function LSTM:reset(stdv)
   --for weight_x
   local stdv_x, stdv_h
   if stdv then
      stdv_x = stdv * math.sqrt(3)
      stdv_x = stdv_h
   else
      stdv_x = 1./math.sqrt(self.weight_x:size(2))
      stdv_h = 1./math.sqrt(self.weight_h:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight_x:size(1) do
         self.weight_x:select(1, i):apply(function()
            return torch.uniform(-stdv_x, stdv_x)
         end)
      end
   else
      self.weight_x:uniform(-stdv_x, stdv_x)
      if self.bias_x then self.bias_x:uniform(-stdv_x, stdv_x) end
   end
   if nn.oldSeed then
      for i=1,self.weight_h:size(1) do
         self.weight_h:select(1, i):apply(function()
            return torch.uniform(-stdv_h, stdv_h)
         end)
      end
   else
      self.weight_h:uniform(-stdv_h, stdv_h)
      if self.bias_h then self.bias_h:uniform(-stdv_h, stdv_h) end
   end
   return self
end

function LSTM:updateOutput(input)
   batch_size = input[1]:size(1)
   self.output_c:resize(batch_size, self.weight_h:size(1))
   self.output_h:resize(batch_size, self.weight_h:size(1))

   if self.dnnPrimitives then
      self.mkldnnInitOk = 1
   else
      self.mkldnnInitOk = 0
   end
   self.dnnPrimitives = self.dnnPrimitives or torch.LongTensor(30)

   --TODO bias = NULL
   --input is c, h, x
   input[1].THNN.LSTM_updateOutput(
      self.dnnPrimitives:cdata(),
      self.mkldnnInitOk,
      input[1]:cdata(),
      input[2]:cdata(),
      input[3]:cdata(),
      self.output_c:cdata(),
      self.output_h:cdata(),
      self.weight_h:cdata(),
      self.weight_x:cdata(),
      self.bias_h:cdata(),
      self.bias_x:cdata()
   )

   return self.output
end

function LSTM:updateGradInput(input, gradOutput)
   batch_size = input[1]:size(1)
   self.grad_in_c:resize(batch_size, self.weight_h:size(1))
   self.grad_in_h:resize(batch_size, self.weight_h:size(1))
   self.grad_in_x:resize(batch_size, self.weight_x:size(1))
   --TODO bias = NULL
   --input is c, h, x
   --grad output should be c, h
   input[1].THNN.LSTM_updateGradInput(
      self.dnnPrimitives:cdata(),
      input[1]:cdata(),
      input[2]:cdata(),
      input[3]:cdata(),
      self.weight_h:cdata(),
      self.weight_x:cdata(),
      gradOutput[1]:cdata(),  --c
      gradOutput[2]:cdata(),  --h
      self.grad_weight_h:cdata(),
      self.grad_weight_x:cdata(),
      self.grad_bias_h:cdata(),
      self.grad_bias_x:cdata(),
      self.grad_in_c:cdata(),
      self.grad_in_h:cdata(),
      self.grad_in_x:cdata()
   )
   return self.gradInput
end

function LSTM:accGradParameters(input, gradOutput, scale)
end

function LSTM:sharedAccUpdateGradParameters(input, gradOutput, lr)
   -- we do not need to accumulate parameters when sharing:
   self:defaultAccUpdateGradParameters(input, gradOutput, lr)
end

function LSTM:parameters()
   if self.bias_h then
      return {self.weight_x, self.bias_x, self.weight_h, self.bias_h},
             {self.grad_weight_x, self.grad_bias_x, self.grad_weight_h, self.grad_bias_h}
   else
      return {self.weight_x, self.weight_h}, {self.grad_weight_x, self.grad_weight_h}
   end
end


function LSTM:clearState()
   return parent.clearState(self)
end

function LSTM:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight_x:size(2), self.weight_x:size(1)) ..
      (self.bias == nil and ' without bias' or '')
end
