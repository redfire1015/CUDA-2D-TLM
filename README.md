# CUDA-2D-TLM
A CUDA implementation of 2D-TLM (Transmission Line Method).

## Summary
1. The program is broken down into 4 stages, source, scatter, connect and boundry.
2. Each of these stages can be completed in parralell allowing for a massive cumulative speedup.
3. A full writeup of TLM and its applications can be found [Here](https://en.wikipedia.org/wiki/Transmission-line_matrix_method)

# Timings:
<p align=center>
<img width="303" alt="image" src="https://github.com/user-attachments/assets/6de18ecb-43fc-416c-b67b-703f34fb2ec6">
</p>
<p align=center>
Graph of relative speedup when comparing run times between the CPU and GPU versions of the 2D TLM algorithm for varying time steps.
</p>

<p align=center>
<img width="296" alt="image" src="https://github.com/user-attachments/assets/fe5fbfbb-9b01-4cf7-b9a0-3f654f85f5c4">
</p>
<p align=center>
Graph of relative speedup when comparing run times between the CPU and GPU versions of the 2D TLM algorithm for varying grid sizes.
</p>
