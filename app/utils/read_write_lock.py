"""
Async read-write lock implementation for KeyManager.

Allows multiple concurrent readers or a single exclusive writer.
Implements write-preferring policy: waiting writers get priority over new readers.
This ensures readers always get the latest data and prevents writer starvation.
"""
import asyncio
from contextlib import asynccontextmanager


class ReadWriteLock:
    """
    Async read-write lock with write-preferring policy.
    
    Multiple readers can acquire the lock simultaneously.
    Writers get exclusive access (no readers or other writers).
    
    Write-Preferring Policy:
    - When writers are waiting, new readers are blocked to give writers priority
    - This ensures readers get the latest data and prevents writer starvation
    - Active readers are allowed to finish, but new readers must wait
    
    Example:
        lock = ReadWriteLock()
        
        # Multiple reads can happen concurrently (if no writers waiting)
        async with lock.read_lock():
            data = await fetch_data()
        
        # Write has exclusive access and gets priority when waiting
        async with lock.write_lock():
            await update_data()
    """
    
    def __init__(self):
        """Initialize the read-write lock with write-preferring policy."""
        # Internal lock for protecting reader/writer state
        self._lock = asyncio.Lock()
        # Condition variable for waiting readers/writers
        self._condition = asyncio.Condition(self._lock)
        # Number of active readers
        self._readers = 0
        # Number of writers waiting (priority tracking)
        self._writers_waiting = 0
        # Whether a writer currently has the lock
        self._writer_active = False
        
    async def acquire_read(self) -> None:
        """
        Acquire read lock - multiple readers allowed.
        
        Blocks if writers are active or waiting (write-preferring policy).
        This ensures readers get the latest data by letting writers update first.
        
        Priority order:
        1. If writer is active: wait for writer to finish
        2. If writers are waiting: wait for writers to complete first (priority)
        3. Otherwise: allow concurrent reading
        """
        async with self._lock:
            # Write-preferring: block new readers if writers are waiting or active
            # This gives priority to writers so readers get latest data
            while self._writer_active or self._writers_waiting > 0:
                await self._condition.wait()
            # No writers active or waiting - safe to read
            self._readers += 1
            
    async def release_read(self) -> None:
        """
        Release read lock.
        
        Notifies waiting writers when all readers finish (write priority).
        Writers are prioritized over new readers.
        """
        async with self._lock:
            self._readers -= 1
            # When all readers finish, notify waiting writers (priority)
            if self._readers == 0:
                # Writers get priority - notify them first
                self._condition.notify_all()
                
    async def acquire_write(self) -> None:
        """
        Acquire write lock - exclusive access with priority.
        
        Blocks until no readers or writers are active.
        Once a writer starts waiting, new readers are blocked (priority).
        """
        async with self._lock:
            # Increment waiting count first (signals priority to readers)
            self._writers_waiting += 1
            try:
                # Wait until no readers or writers are active
                # New readers will see _writers_waiting > 0 and block
                while self._readers > 0 or self._writer_active:
                    await self._condition.wait()
                # Mark writer as active - blocks all readers
                self._writer_active = True
            finally:
                # Decrement waiting count after acquiring lock
                self._writers_waiting -= 1
                
    async def release_write(self) -> None:
        """
        Release write lock.
        
        Notifies all waiting readers and writers.
        Writers get processed first due to write-preferring policy.
        """
        async with self._lock:
            self._writer_active = False
            # Notify all waiters - writers will acquire first due to priority check
            self._condition.notify_all()
            
    @asynccontextmanager
    async def read_lock(self):
        """
        Context manager for read lock.
        
        Usage:
            async with lock.read_lock():
                # Multiple readers can be here simultaneously
                await read_operation()
        """
        await self.acquire_read()
        try:
            yield
        finally:
            await self.release_read()
            
    @asynccontextmanager
    async def write_lock(self):
        """
        Context manager for write lock.
        
        Usage:
            async with lock.write_lock():
                # Exclusive access - no other readers or writers
                await write_operation()
        """
        await self.acquire_write()
        try:
            yield
        finally:
            await self.release_write()
            
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"ReadWriteLock(readers={self._readers}, "
            f"writers_waiting={self._writers_waiting}, "
            f"writer_active={self._writer_active})"
        )
    
    def get_stats(self) -> dict:
        """
        Get current lock statistics for monitoring.
        
        Returns:
            dict: Statistics including active readers, waiting writers, etc.
        """
        return {
            "active_readers": self._readers,
            "writers_waiting": self._writers_waiting,
            "writer_active": self._writer_active,
        }

