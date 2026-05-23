use std::ffi::c_void;
use std::io;
use std::os::fd::RawFd;

use crate::error::MoEError;

/// Unified expert weight storage: either raw packed or LZ4-compressed.
pub enum ExpertFile {
    Raw { fd: RawFd, expert_size: usize },
    Lz4 { fd: RawFd, offsets: Vec<u32>, expert_size: usize },
}

impl ExpertFile {
    pub fn expert_size(&self) -> usize {
        match self {
            ExpertFile::Raw { expert_size, .. } => *expert_size,
            ExpertFile::Lz4 { expert_size, .. } => *expert_size,
        }
    }

    pub fn fd(&self) -> RawFd {
        match self {
            ExpertFile::Raw { fd, .. } => *fd,
            ExpertFile::Lz4 { fd, .. } => *fd,
        }
    }

    /// Read the weights for a single expert into `dst` (must be expert_size bytes).
    pub fn read_expert(&self, expert_idx: usize, dst: &mut [u8]) -> Result<(), MoEError> {
        match self {
            ExpertFile::Raw { fd, expert_size } => {
                let off = (expert_idx as i64) * (*expert_size as i64);
                let n = unsafe {
                    libc::pread(*fd, dst.as_mut_ptr() as *mut c_void, *expert_size, off)
                };
                if n != *expert_size as isize {
                    Err(MoEError::Io(io::Error::new(io::ErrorKind::UnexpectedEof,
                        format!("pread expert {}: got {} expected {}", expert_idx, n, expert_size))))
                } else {
                    Ok(())
                }
            }
            ExpertFile::Lz4 { fd, offsets, expert_size } => {
                let n_experts = offsets.len().saturating_sub(1);
                let hdr_bytes = (n_experts + 2) * 4; // [num_experts] + N+1 offsets
                let comp_off = offsets[expert_idx] as usize;
                let comp_end = offsets[expert_idx + 1] as usize;
                let comp_sz = comp_end - comp_off;
                let file_off = (hdr_bytes + comp_off) as i64;
                let mut comp = vec![0u8; comp_sz];
                let n = unsafe {
                    libc::pread(*fd, comp.as_mut_ptr() as *mut c_void, comp_sz, file_off)
                };
                if n != comp_sz as isize {
                    return Err(MoEError::Io(io::Error::new(io::ErrorKind::UnexpectedEof,
                        format!("pread lz4 expert {}: got {} expected {}", expert_idx, n, comp_sz))));
                }
                let decomp = lz4_flex::decompress(&comp, *expert_size)
                    .map_err(|e| MoEError::Io(io::Error::new(io::ErrorKind::InvalidData,
                        format!("lz4 decompress expert {}: {}", expert_idx, e))))?;
                let n = (*expert_size).min(decomp.len());
                dst[..n].copy_from_slice(&decomp[..n]);
                Ok(())
            }
        }
    }
}

impl Drop for ExpertFile {
    fn drop(&mut self) {
        unsafe { libc::close(self.fd()); }
    }
}
