/**
 * WebGPU Context Management
 */

/// <reference types="@webgpu/types" />

export class WebGPUContext {
  private device: GPUDevice | null = null;
  private adapter: GPUAdapter | null = null;

  /**
   * Initialize WebGPU context
   */
  async initialize(): Promise<void> {
    if (!navigator.gpu) {
      throw new Error('WebGPU is not supported in this browser');
    }

    this.adapter = await navigator.gpu.requestAdapter();
    if (!this.adapter) {
      throw new Error('Failed to get WebGPU adapter');
    }

    this.device = await this.adapter.requestDevice();
    if (!this.device) {
      throw new Error('Failed to get WebGPU device');
    }

    // Handle device lost (if supported)
    if (this.device.lost) {
      this.device.lost.then(info => {
        console.error('WebGPU device lost:', info.message);
      });
    }
  }

  /**
   * Get the WebGPU device
   */
  getDevice(): GPUDevice {
    if (!this.device) {
      throw new Error('WebGPU context not initialized');
    }
    return this.device;
  }

  /**
   * Get the WebGPU adapter
   */
  getAdapter(): GPUAdapter {
    if (!this.adapter) {
      throw new Error('WebGPU context not initialized');
    }
    return this.adapter;
  }

  /**
   * Create a buffer with initial data
   */
  createBuffer(
    data: Float64Array | Float32Array,
    usage: GPUBufferUsageFlags = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  ): GPUBuffer {
    const device = this.getDevice();
    const buffer = device.createBuffer({
      size: data.byteLength,
      usage,
      mappedAtCreation: true,
    });

    if (data instanceof Float64Array) {
      new Float64Array(buffer.getMappedRange()).set(data);
    } else {
      new Float32Array(buffer.getMappedRange()).set(data);
    }
    buffer.unmap();

    return buffer;
  }

  /**
   * Create an output buffer
   */
  createOutputBuffer(size: number): GPUBuffer {
    const device = this.getDevice();
    return device.createBuffer({
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
  }

  /**
   * Create a staging buffer for reading results
   */
  createStagingBuffer(size: number): GPUBuffer {
    const device = this.getDevice();
    return device.createBuffer({
      size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  /**
   * Read buffer data back to CPU
   */
  async readBuffer(buffer: GPUBuffer, size: number): Promise<ArrayBuffer> {
    const device = this.getDevice();
    const stagingBuffer = this.createStagingBuffer(size);

    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, size);
    device.queue.submit([commandEncoder.finish()]);

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const data = stagingBuffer.getMappedRange().slice(0);
    stagingBuffer.unmap();
    stagingBuffer.destroy();

    return data;
  }

  /**
   * Cleanup resources
   */
  destroy(): void {
    if (this.device) {
      this.device.destroy();
      this.device = null;
    }
    this.adapter = null;
  }
}

// Global context instance
let globalContext: WebGPUContext | null = null;

/**
 * Get or create the global WebGPU context
 */
export async function getWebGPUContext(): Promise<WebGPUContext> {
  if (!globalContext) {
    globalContext = new WebGPUContext();
    await globalContext.initialize();
  }
  return globalContext;
}












