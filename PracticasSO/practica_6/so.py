#!/usr/bin/env python

from hardware import *
import log
import math

## emulates a compiled program
class Program():

    def __init__(self, name, instructions):
        self._name = name
        self._instructions = self.expand(instructions)

    @property
    def name(self):
        return self._name

    @property
    def instructions(self):
        return self._instructions

    def addInstr(self, instruction):
        self._instructions.append(instruction)

    def expand(self, instructions):
        expanded = []
        for i in instructions:
            if isinstance(i, list):
                ## is a list of instructions
                expanded.extend(i)
            else:
                ## a single instr (a String)
                expanded.append(i)

        ## now test if last instruction is EXIT
        ## if not... add an EXIT as final instruction
        last = expanded[-1]
        if not ASM.isEXIT(last):
            expanded.append(INSTRUCTION_EXIT)

        return expanded

    def __repr__(self):
        return "Program({name}, {instructions})".format(name=self._name, instructions=self._instructions)

#Ejercicio 6.2 implentar
class Loader():

    def __init__(self, fileSystem, kernel):
        self._free = 0
        self._fileSystem = fileSystem
        self._kernel = kernel

    def load(self, pcb):
        instructions = self._kernel.fileSystem.read(pcb.path)
        progSize = len(instructions)
        numberOfPages = self.calculatePages(progSize)
        paginated = self._kernel.memoryManager.paging(instructions, numberOfPages)
        pageTable = PageTable(numberOfPages, paginated)
        self._kernel.memoryManager.putPageTable(pcb.pid, pageTable)

    def loadPage(self, pageId, pagetable):
        if (pagetable.isSwaped(pageId)):
            runningPcb = self._kernel.pcbtable.runningPCB
            key = "pid: " + str(runningPcb.pid) + " pageId: " + str(pageId)
            instructions = self._kernel.swap.swapIn(key, pagetable, pageId)
        else:
            instructions = pagetable.getPage(pageId)
        frameSize = self._kernel.memoryManager.frameSize
        frameId = self._kernel.memoryManager.allocFrames(1)[0]
        frameBaseDir = frameSize * frameId
        pagetable.assignFrame(pageId, frameId)
        pagetable.setValidBit(pageId, 1)
        for offset in range(0, len(instructions)):
            physicalAddress = frameBaseDir + offset
            HARDWARE.memory.write(physicalAddress, instructions[offset])


    def calculatePages(self, progSize):
        frameSize = self._kernel.memoryManager.frameSize
        return math.ceil(progSize / frameSize)

class PageTable():
    def __init__(self, pages, paginatedPrg):
        self._len = 0
        self._validBits = []
        self._swapedBits = []
        self._table = self.build(pages)
        self._paginatedPrg = paginatedPrg

    def build(self, numberOfPages):
        pageTable = dict()
        for i in range(0, numberOfPages):
            pageTable[i] = None
            self._validBits.append(0)
            self._swapedBits.append(0)
            self.len += 1
        return pageTable

    @property
    def len(self):
        return self._len

    @len.setter
    def len(self, newLen):
        self._len = newLen

    def lastPageId(self):
        return self.len - 1

    def getPage(self, pageId):
        return self._paginatedPrg[pageId]

    def assignFrame(self, pageId, frameId):
        self._table[pageId] = frameId

    def getFrameFor(self, pageId):
        return self._table[pageId]

    def allAssingnedFrames(self):
        return self._table.values()

    def allPages(self):
        return len(self._table.keys())

    def setValidBit(self, pageId, value):
        self._validBits[pageId] = value

    def isInMemory(self, pageId):
        return self._validBits[pageId] == 1

    def setSwapedBit(self, pageId, value):
        self._swapedBits[pageId] = value

    def isSwaped(self, pageId):
        return self._swapedBits[pageId] == 1


class Dispatcher():

    def __init__(self, memoryManager):
        self._memoryManager = memoryManager

    def save(self, pcb):
        pcb.pc = HARDWARE.cpu.pc
        HARDWARE.cpu.pc = -1
        log.logger.info("Dispatcher save {pId}".format(pId = pcb.pcbId))

    def load(self, pcb):
        HARDWARE.cpu.pc = pcb.pc
        HARDWARE.timer.reset()
        self._memoryManager.initializeMMUFor(pcb)
        log.logger.info("Dispatcher load {pId}".format(pId = pcb.pcbId))

    @property
    def memoryManager(self):
        return self._memoryManager

class FileSystem():

    def __init__(self):
        self._disk = dict()

    def write(self, path, instructions):
        self._disk[path] = self.expand(instructions)

    def read(self, path):
        return self._disk[path]

    def getSize(self, path):
        return len(self._disk[path])

    def expand(self, instructions):
        expanded = []
        for i in instructions:
            if isinstance(i, list):
                ## is a list of instructions
                expanded.extend(i)
            else:
                ## a single instr (a String)
                expanded.append(i)

        ## now test if last instruction is EXIT
        ## if not... add an EXIT as final instruction
        last = expanded[-1]
        if not ASM.isEXIT(last):
            expanded.append(INSTRUCTION_EXIT)

        return expanded

class Dispatcher():

    def load(self, pcb, pageTablePCB):
        HARDWARE.cpu.pc = pcb.pc
        HARDWARE.timer.reset()
        self.setAllPages(pageTablePCB)

    def save(self, pcb):
        pcb.pc = HARDWARE.cpu.pc
        HARDWARE.cpu.pc = -1

    def setAllPages(self, pageTablePCB):
        pages = pageTablePCB.allPages()
        HARDWARE.mmu.resetTLB()
        for page in range(0, pages):
            self.setPage(page, pageTablePCB)

    def setPage(self, pageId, pageTablePCB):
        assignedFrame = pageTablePCB.getFrameFor(pageId)
        HARDWARE.mmu.setPageFrame(pageId, assignedFrame)

class DefaultMemoryManager():

    def initializeMMUFor(self, pcb):
        HARDWARE.mmu.baseDir = pcb.baseDir

    def dispose(self, pcb):
        return False

class MemoryManager():
    def __init__(self):
        self._frameSize = HARDWARE.mmu.frameSize
        self._frames = self.buildFrameList()
        self._pages = []

    @property
    def frameSize(self):
        return self._frameSize

    def hasFreeFrames(self):
        return not (len(self._frames) == 0)

    def buildFrameList(self):
        size = HARDWARE.memory.size
        frameSize = self._frameSize
        listSize = math.ceil(size / frameSize)
        toReturn = []
        for i in range(0, listSize):
            toReturn.append(i)
        return toReturn

    def allocFrames(self, numberOfFrames):
        framesToReturn = self._frames[0:numberOfFrames]
        for i in range(0, numberOfFrames):
            self._frames.pop(0)
        return framesToReturn

    def freeFrames(self, listOfFrames):
        for frame in listOfFrames:
            if frame is not None:
                self._frames.append(frame)

    def putPageTable(self, pid, pageTable):
        self._pages[pid] = pageTable

    def getPageTable(self, pid):
        return self._pages[pid]

    def paging(self, instructions, pages):
        frameSize = self.frameSize
        progSize = len(instructions)
        paginated = []
        for i in range(0, pages):
            paginated.append([])
        for logicalAddress in range(0, progSize):
            pageId =logicalAddress // frameSize
            paginated[pageId].append(instructions[logicalAddress])
        return paginated

class Swap():

    def __init__(self):
        self._disk = dict()

    def write(self, id, instructions):
        self._disk[id] = instructions

    def read(self, id):
        return self._disk[id]

    def swapIn(self, key, pageTable, pageId):
        toReturn = self._disk[key]
        pageTable.setSwapedBit(pageId, 0)
        self._disk.pop(key)
        return toReturn

    def swapOut(self, pId, pageTable, pageId ):
       frameSize = HARDWARE.mmu.frameSize
       frameId = pageTable.getFrameFor(pageId)
       frameBaseDir = frameSize + frameId
       page = pageTable.getPage(pageId)
       pageSize = len(page)
       toSave = []
       for offset in range(0, pageSize):
            physicalAddress = frameBaseDir + offset
            insInMemory = HARDWARE.memory.read(physicalAddress)
            toSave.append(insInMemory)
       pageTable.assignFrame(pageId, None)
       pageTable.setValidBit(pageId, 0)
       pageTable.setSwapedBit(pageId, 1)
       self._disk["pid: " + str(pId) + " pageId: " + str(pageId)] = toSave

class FIFOVictimSelectionAlgorithm():
    def __init__(self):
        self._pagesQueue = []

    def selectVictim(self):
        toReturn = self._pagesQueue.pop(0)
        while not toReturn[2].isInMemory(toReturn[1]):
            toReturn = self._pagesQueue.pop(0)
        return toReturn

    def addReference(self, pid, pageId, pagetable):
        referencePair = (pid, pageId, pagetable)
        self._pagesQueue.append(referencePair)


class PCB():

    def __init__(self, pid, path, prioridad = None):
        self._pid = pid
        self._path = path
        self._status = State.New
        self._baseDir = 0
        self._limit = 0
        self._pc = 0
        self._prioridad = prioridad

    @property
    def path(self):
        return self._path

    @property
    def pid(self):
        return self._pid

    @property
    def prioridad(self):
        return self._prioridad

    @prioridad.setter
    def prioridad(self, newPriority):
        self._prioridad = newPriority

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value):
        self._status = value

    @property
    def baseDir(self):
        return self._baseDir

    @property
    def limit(self):
        return self._limit

    @property
    def pc(self):
        return self._pc

    @pc.setter
    def pc(self, value):
        self._pc = value

    def __repr__(self):
        return "{path}".format(path=self.path)

class State():

    New = "new"
    Running = "running"
    Ready = "ready"
    Waiting = "waiting"
    Terminated = "terminated"

class PCBTable():

    def __init__(self):
        self._pcb_table = []
        self._running = None
        self._currentPID = 0


    def createPcb (self, path, prioridad):
        pId = len(self.pcbTable)
        pcb = PCB(pId, path, prioridad)
        self.pcbTable.append(pcb)
        return pcb


    def hasRunning(self):
        return not (self.running == None)

    @property
    def pcbTable(self):
        return self._pcb_table

    @property
    def running(self):
        return self._running

    @running.setter
    def running(self, value):
        self._running = value

    @property
    def currentPID(self):
        return self._currentPID

    @currentPID.setter
    def currentPID(self, pid):
        self._currentPID = pid

    def get(self, pid):
        return self.pcbTable[pid]

    def add(self, pcb):
        pcbPid = pcb.pid
        self.pcbTable[pcbPid] = pcb

    def remove(self, pid):
        self.pcbTable.pop(pid)

    def getNewPID(self):
        newPid = self._currentPID
        self._currentPID += 1
        return newPid

class SchedulerFCFS():

    def __init__(self):
        self._readyqueue = []

    def add(self, pcb):
        self.readyQueue.append(pcb)

    def isEmpty(self):
        return len(self.readyQueue) == 0

    @property
    def readyQueue(self):
        return self._readyqueue

    @readyQueue.setter
    def readyQueue(self,value):
        self._readyqueue = value

    def getNext(self):
        if self.isEmpty():
            return None
        else:
            return self.readyQueue.pop(0)

    def debeExpropiar(self, PCBruning, PCBentrante):
        return False

class SchedulerPrioridadExpropiativo(SchedulerFCFS):

    def debeExpropiar(self, PCBruning, PCBentrante):
        return PCBruning.prioridad < PCBentrante.prioridad

    def add(self, pcb):
        listaNueva = []
        newAgedPCB = AgedPCB(pcb)
        while not self.isEmpty() and (self.readyQueue[0].auxPriority > newAgedPCB.auxPriority):
            listaNueva.append(self.readyQueue.pop(0))
        listaNueva.append(newAgedPCB)
        # self._readyqueue = listaNueva + self.readyQueue
        self.readyQueue = listaNueva + self.readyQueue

    def getNext(self):
        if self.isEmpty():
            return None
        else:
            self.readyQueue = list(map(lambda aged: aged.addPriority(), self.readyQueue))
            return self.readyQueue.pop(0).pcb

class SchedulerPrioridadNoExpropiativo(SchedulerPrioridadExpropiativo):

    def __init__(self):
        super().__init__()

    def debeExpropiar(self, PCBruning, PCBentrante):
        return False

class SchedulerRoundRobin(SchedulerFCFS):

    def __init__(self, quantum):
        super().__init__()
        HARDWARE.timer.quantum = quantum

class AgedPCB():

    def __init__(self, pcb):
        self._pcb = pcb
        self._auxPriority = pcb.prioridad

    @property
    def pcb(self):
        return self._pcb

    @property
    def auxPriority(self):
        return self._auxPriority

    @auxPriority.setter
    def auxPriority(self, value):
        self._auxPriority = value

    def addPriority(self):
        self.auxPriority = self.auxPriority + 1
        return self

#Algoritmo de selección de víctima
class Victim:
    def __init__(self, pId, page):
        self._pId = pId
        self._page = page

    @property
    def pId(self):
        return self._pId

    @property
    def page(self):
        return self._page



## emulates an Input/Output device controller (driver)
class IoDeviceController():

    def __init__(self, device):
        self._device = device
        self._waiting_queue = []
        self._currentPCB = None

    def runOperation(self, pcb, instruction):
        pair = {'pcb': pcb, 'instruction': instruction}
        # append: adds the element at the end of the queue
        self._waiting_queue.append(pair)
        # try to send the instruction to hardware's device (if is idle)
        self.__load_from_waiting_queue_if_apply()

    def getFinishedPCB(self):
        finishedPCB = self._currentPCB
        self._currentPCB = None
        self.__load_from_waiting_queue_if_apply()
        return finishedPCB

    def __load_from_waiting_queue_if_apply(self):
        if (len(self._waiting_queue) > 0) and self._device.is_idle:
            ## pop(): extracts (deletes and return) the first element in queue
            pair = self._waiting_queue.pop(0)
            #print(pair)
            pcb = pair['pcb']
            instruction = pair['instruction']
            self._currentPCB = pcb
            self._device.execute(instruction)


    def __repr__(self):
        return "IoDeviceController for {deviceID} running: {currentPCB} waiting: {waiting_queue}".format(deviceID=self._device.deviceId, currentPCB=self._currentPCB, waiting_queue=self._waiting_queue)


## emulates the  Interruptions Handlers
class AbstractInterruptionHandler():
    def __init__(self, kernel):
        self._kernel = kernel

    @property
    def kernel(self):
        return self._kernel

    def execute(self, irq):
        log.logger.error("-- EXECUTE MUST BE OVERRIDEN in class {classname}".format(classname=self.__class__.__name__))

    def putRunningSiDebe(self, pcb):
        pcb_running = self.kernel.pcbTable.hasRunning
        if (HARDWARE.cpu.pc == -1):  #idle
            self.putRunning(pcb)
        elif self.kernel.scheduler.mustExpropiate(pcb_running, pcb):
            self.expropiate()
            self.putRunning(pcb)
        else:
            pcb.state = State.Ready
            self.kernel.scheduler.add(pcb)

    def putRunning(self, pcbToRun):
        pcbToRun.state = State.Running
        pageTable = self.kernel.memoryManager.getPageTable(pcbToRun.pid)
        self.kernel.pcbtable.runningPCB = pcbToRun
        self.kernel.dispatcher.load(pcbToRun, pageTable)

    def expropiate(self):
        pcbToExpropiate = self.kernel.pcbtable.runningPCB
        self.quit_running_with_state(State.Ready)
        self.kernel.scheduler.add(pcbToExpropiate)

    def quit_running_with_state(self, state):
        pcbToQuit = self.kernel.pcbTable.hasRunning
        pcbToQuit.state = state
        self.kernel.dispatcher.save(pcbToQuit)
#Las 4 interrupciones:

#NEW
class NewInterruptionHandler(AbstractInterruptionHandler):

    def execute(self, irq):
        path = irq.parameters[0] #"path"
        priority = irq.parameters[1] #"priority"
        newPid = self.kernel.pcbTable.getNewPID()
        pcb = PCB(path, newPid, priority)
        self.putRunningSiDebe(pcb)

#KIL
class KillInterruptionHandler(AbstractInterruptionHandler):

    def execute(self, irq):
        running = self.kernel.pcbTable.running
        self.kernel.pcbTable.running = None

        self.kernel.dispatcher.save(running)
        running.status = State.Terminated

        self.kernel.memoryManager.dispose(running)

        nextPCB = self.kernel.scheduler.getNext()
        if (not nextPCB is None):
            nextPCB.status = State.Running
            self.kernel.pcbTable.running = nextPCB
            self.kernel.dispatcher.load(nextPCB)

#IOIN
class IoInInterruptionHandler(AbstractInterruptionHandler):

    def execute(self, irq):
        running = self.kernel.pcbTable.running
        self.kernel.dispatcher.save(running)
        running.status = State.Waiting
        self.kernel.pcbTable.running = None
        self.kernel.ioDeviceController.runOperation(running, irq)
        nextPCB = self.kernel.scheduler.getNext()
        if not nextPCB is None:
            nextPCB.status = State.Running
            self.kernel.pcbTable.running = nextPCB
            self.kernel.dispatcher.load(nextPCB)

#IOOUT
class IoOutInterruptionHandler(AbstractInterruptionHandler):

    def execute(self, irq):
        pcb = self.kernel.ioDeviceController.getFinishedPCB()
        log.logger.info("------{x}".format(x= self.kernel.pcbTable.hasRunning()))
        running = self.kernel.pcbTable.running

        if not self.kernel.pcbTable.hasRunning():
            pcb.status = State.Running
            self.kernel.pcbTable.running = pcb
            self.kernel.dispatcher.load(pcb)
        else:
            if self.kernel.scheduler.debeExpropiar(running, pcb):
                self.kernel.pcbTable.running = None

                self.kernel.dispatcher.save(running)
                running.status = State.Ready
                self.kernel.scheduler.add(running)

                pcb.status = State.Running
                self.kernel.pcbTable.running = pcb
                self.kernel.dispatcher.load(pcb)
            else:
                pcb.state = State.Ready
                self.kernel.scheduler.add(pcb)

#TIMEOUT
class TimeoutInterruptionHandler(AbstractInterruptionHandler):

    def execute(self, irq):
        running = self.kernel.pcbTable.running

        self.kernel.pcbTable.running = None

        self.kernel.dispatcher.save(running)
        running.status = State.Ready
        self.kernel.scheduler.add(running)

        nextPcb = self.kernel.scheduler.getNext()
        if(not nextPcb is None):
            self.kernel.pcbTable.running = nextPcb
            nextPcb.status = State.Running
            self.kernel.dispatcher.load(nextPcb)

class PageFaultInterruptionHandler(AbstractInterruptionHandler):

#    def execute(self, irq):
#        page = irq.parameters
#        running = self.kernel.pcbTable.running
#        self.kernel.loader.loadPage(running, page)

    def execute(self, irq):
        if not self.kernel.memoryManager.hasFreeFrames():
            victimPair = self.kernel.victimSelectionAlgorithm.selectVictim()
            victimPid = victimPair[0]
            victimPageId = victimPair[1]
            victimPageTable = victimPair[2]
            frameToUse = victimPageTable.getFrameFor(victimPageId)
            self.kernel.swap.swapOut(victimPid, victimPageTable, victimPageId)
            self.kernel.memoryManager.freeFrames([frameToUse])
        pageId = irq.parameters
        runningPcb = self.kernel.pcbtable.runningPCB
        pagetable = self.kernel.memoryManager.getPageTable(runningPcb.pid)
        self.kernel.loader.loadPage(pageId, pagetable)
        self.kernel.dispatcher.setPage(pageId, pagetable)
        self.kernel.victimSelectionAlgorithm.addReference(runningPcb.pid, pageId, pagetable)

# emulates the core of an Operative System
class Kernel():

    def __init__(self):
        ## setup interruption handlers
        killHandler = KillInterruptionHandler(self)
        HARDWARE.interruptVector.register(KILL_INTERRUPTION_TYPE, killHandler)

        ioInHandler = IoInInterruptionHandler(self)
        HARDWARE.interruptVector.register(IO_IN_INTERRUPTION_TYPE, ioInHandler)

        ioOutHandler = IoOutInterruptionHandler(self)
        HARDWARE.interruptVector.register(IO_OUT_INTERRUPTION_TYPE, ioOutHandler)

        newHandler = NewInterruptionHandler(self)
        HARDWARE.interruptVector.register(NEW_INTERRUPTION_TYPE, newHandler)

        timeOutHandler = TimeoutInterruptionHandler(self)
        HARDWARE.interruptVector.register(TIMEOUT_INTERRUPTION_TYPE, timeOutHandler)

        pageFaultHandler = PageFaultInterruptionHandler(self)
        HARDWARE.interruptVector.register(PAGE_FAULT_INTERRUPTION_TYPE, pageFaultHandler)

        ## controls the Hardware's I/O Device
        self._ioDeviceController = IoDeviceController(HARDWARE.ioDevice)
        self._dispatcher = Dispatcher()
        self._pcbTable = PCBTable()
        self._interruptVector = HARDWARE.interruptVector
        self._fileSystem = FileSystem()
        self._memoryManager = MemoryManager()
        self._loader = Loader(self._fileSystem, self)
        self._scheduler = SchedulerFCFS()
        self._swap = Swap()
        self._victimSelectionAlgorithm = FIFOVictimSelectionAlgorithm()


    @property
    def memoryManager(self):
        return self._memoryManager

    @property
    def ioDeviceController(self):
        return self._ioDeviceController

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def dispatcher(self):
        return self._dispatcher

    @dispatcher.setter
    def dispatcher(self, dispatcher):
        self._dispatcher = dispatcher

    @property
    def pcbTable(self):
        return self._pcbTable

    @property
    def loader(self):
        return self._loader

    @property
    def fileSystem(self):
        return self._fileSystem

    @property
    def swap(self):
        return self._swap

    @property
    def victimSelectionAlgoritm(self):
        return self._victimSelectionAlgorithm
#6.1: Implementar la interrupción #NEW

    ## emulates a "system call" for programs execution
    def run(self, path, priority):
        newIRQ = IRQ(NEW_INTERRUPTION_TYPE, parameters=(path, priority))
        self._interruptVector.handle(newIRQ)


    def __repr__(self):
        return "Kernel "

